"""
Combined approach:
1. Deterministic Rank-65 baseline via Kronecker-embedded Strassen (4x4x4) + Naive (4x4x1).
2. "Linear algebraic peeling": iterative rank-1 tensor extraction via truncated SVD of
   the mode-1 flattening, with each extracted component immediately rounded to the nearest
   half-integer and its exact contribution subtracted from the residual tensor. This avoids
   the local minima of continuous optimization and directly enforces exactness.
3. Exact local-solve polish to tighten the final decomposition.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64 and einsum."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    if Ua.ndim != 2 or Va.ndim != 2 or Wa.ndim != 2:
        return False
    rank = Ua.shape[1]
    if Ua.shape[0] != n*m or Va.shape[0] != m*p or Wa.shape[0] != p*n:
        return False
    if Va.shape[1] != rank or Wa.shape[1] != rank:
        return False

    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)

    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
    return bool(np.allclose(T, 1.0, atol=atol))

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """M is a numpy array or list of lists. Returns list of lists of floats."""
    if isinstance(M, np.ndarray):
        M = M.tolist()
    return [[_round_to_half_integer(x) for x in row] for row in M]

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j ; V index: j*2+k ; W index: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()

    # 1. 4x4x4 part (k=0..3): Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)

            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2

            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2

            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2

            add_term(u, v, w)

    # 2. 4x4x1 part (k=4): Naive
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _residual_tensor(U, V, W, n, m, p):
    """Compute the current decomposition tensor minus the target (all-ones) tensor."""
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
    return T - 1.0

def _peel_residual(U, V, W, n, m, p, deadline, max_terms=100):
    """
    Linear algebraic peeling: repeatedly extract a rank-1 component from the residual
    tensor via the mode-1 flattening SVD, round the three factors to the nearest
    half-integer, and subtract the exact rounded rank-1 contribution from the residual.
    Returns (U, V, W) as float arrays if the residual is fully eliminated within budget,
    else None.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]

    residual = _residual_tensor(U, V, W, n, m, p)

    for _ in range(max_terms):
        if time.time() > deadline:
            return None

        # Stop when residual is (numerically) zero -> valid decomposition found
        if np.max(np.abs(residual)) < 1e-12:
            return U, V, W

        # Mode-1 flattening: shape (n*m, p*n). Residual[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
        # Treat the (i,j) axis as the row. For a rank-1 extraction, find a left vector a (n*m,)
        # such that the residual can be written as a[i,j] * B[i,j,k] where B is the remaining
        # (j,k) structure. Use SVD of the flattened residual.
        flat = residual.reshape(n*m, p*n)  # rows = (i,j), cols = (k,i)

        try:
            U_svd, s, Vt_svd = np.linalg.svd(flat, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        # Take the dominant singular direction.
        if s.size == 0 or s[0] < 1e-12:
            # Residual is effectively zero already.
            return U, V, W

        u_vec = U_svd[:, 0]          # (n*m,)
        sigma = s[0]
        v_vec = Vt_svd[0, :]         # (p*n,)

        # The dominant rank-1 approximation of the residual is:
        #   R[i,j,k] ~= sigma * u_vec[i,j] * v_vec[k,i]
        # We want a rank-1 tensor u_ext[i,j]*v_ext[j,k]*w_ext[k,i] matching this.
        # Factor the middle: v_vec is indexed by (k,i) but we need v_ext[j,k]*w_ext[k,i].
        # The residual's (j,k) coupling is not directly in v_vec. Instead, build the
        # candidate rank-1 by projecting: set u_ext = u_vec, and solve for v_ext, w_ext
        # so that u_ext[i,j]*v_ext[j,k]*w_ext[k,i] approximates R.
        #
        # Simpler robust approach: pick a single (j,k) "pivot" and build a rank-1 term
        # supported on that (j,k) slice, then round. This guarantees exact subtraction.
        #
        # We choose the (i,j,k) entry with the largest magnitude of the residual and
        # construct a rank-1 term that exactly cancels a structured portion.

        # Build a rank-1 term anchored on the dominant left/right singular vectors:
        # Let u_ext[i,j] = u_vec[i,j]. For the (j,k) and (k,i) factors, we use the
        # fact that R[i,j,k] = u_vec[i,j] * v_vec[k,i] * sigma. We need to split
        # v_vec[k,i]*sigma into v_ext[j,k]*w_ext[k,i]. This is underdetermined; instead
        # we use a least-squares split over a fixed j.
        #
        # To keep it exact and simple, we extract via a greedy single-entry anchor:
        # choose (i0,j0,k0) maximizing |R|, then set the rank-1 term to be
        #   u_ext[i,j] = R[i,j,k0] / (v0*w0) etc.
        # But the cleanest exact peeling: use the SVD left/right vectors and solve a
        # 2-factor split by least squares on the (j,k) and (k,i) grids.

        # --- Least-squares split of sigma*u_vec[i,j]*v_vec[k,i] into u_ext*v_ext*w_ext ---
        # Fix u_ext = u_vec. Then we need v_ext[j,k]*w_ext[k,i] ~= sigma*v_vec[k,i] for all (i,k),
        # independent of j, which is impossible unless we let it depend on j. So instead:
        # we treat the residual as a 3-way tensor and do a proper rank-1 extraction using
        # the mode-2 and mode-3 modes.

        # Mode-2 flattening: (m, n*p) -> but residual is 3D. Let's do a direct rank-1
        # extraction via alternating least squares on the residual (a few steps), then
        # round. This is the most reliable within budget.

        # Initialize rank-1 factors from the SVD:
        # u_ext[i,j] from mode-1, then solve v_ext, w_ext by alternating least squares.
        u_ext = u_vec.copy()  # (n*m,)

        # Normalize u_ext
        nu = np.linalg.norm(u_ext)
        if nu < 1e-12:
            return None
        u_ext = u_ext / nu

        # Initialize v_ext, w_ext
        v_ext = np.zeros(m*p)
        w_ext = np.zeros(p*n)
        # seed from v_vec: distribute
        # v_vec is (p*n,) indexed (k,i). w_ext[k,i] ~ v_vec[k,i]; v_ext ~ 1
        w_ext = v_vec.copy()
        nw = np.linalg.norm(w_ext)
        if nw < 1e-12:
            return None
        w_ext = w_ext / nw
        v_ext = np.ones(m*p)

        # Alternating least squares to fit R[i,j,k] = sigma * u_ext[i,j]*v_ext[j,k]*w_ext[k,i]
        target = sigma * u_vec.reshape(n, m)  # (n,m)
        for _als in range(30):
            if time.time() > deadline:
                return None
            # Update v_ext[j,k] = sum_{i} target[i,j]*w_ext[k,i] / sum_i w_ext[k,i]^2
            for j in range(m):
                for k in range(p):
                    num = 0.0
                    den = 0.0
                    for i in range(n):
                        num += target[i, j] * w_ext[k*n+i]
                        den += w_ext[k*n+i] ** 2
                    if den > 1e-12:
                        v_ext[j*p+k] = num / den
            # Update w_ext[k,i] = sum_j target[i,j]*v_ext[j,k] / sum_j v_ext[j,k]^2
            for k in range(p):
                for i in range(n):
                    num = 0.0
                    den = 0.0
                    for j in range(m):
                        num += target[i, j] * v_ext[j*p+k]
                        den += v_ext[j*p+k] ** 2
                    if den > 1e-12:
                        w_ext[k*n+i] = num / den

        # Now we have a real-valued rank-1 term (u_ext, v_ext, w_ext).
        # Round each factor to the nearest half-integer.
        u_ext_r = np.array([_round_to_half_integer(x) for x in u_ext])
        v_ext_r = np.array([_round_to_half_integer(x) for x in v_ext])
        w_ext_r = np.array([_round_to_half_integer(x) for x in w_ext])

        # Compute the exact rounded rank-1 contribution and subtract from residual.
        contrib = np.einsum('ij,jk,ki->ijk',
                            u_ext_r.reshape(n, m),
                            v_ext_r.reshape(m, p),
                            w_ext_r.reshape(p, n),
                            optimize=True)

        # If the rounded contribution is all-zero, this extraction is useless; skip.
        if np.max(np.abs(contrib)) < 1e-12:
            # Try without rounding (keep real) to avoid stalling, but that breaks exactness.
            # Instead, nudge: try the sign-flipped version.
            u_ext_r = -u_ext_r
            v_ext_r = -v_ext_r
            contrib = np.einsum('ij,jk,ki->ijk',
                                u_ext_r.reshape(n, m),
                                v_ext_r.reshape(m, p),
                                w_ext_r.reshape(p, n),
                                optimize=True)
            if np.max(np.abs(contrib)) < 1e-12:
                # Cannot make progress; give up on peeling.
                return None

        # Append the new rank-1 term as a new column.
        new_rank = rank + 1
        U = np.column_stack([U, u_ext_r])
        V = np.column_stack([V, v_ext_r])
        W = np.column_stack([W, w_ext_r])
        rank = new_rank

        # Subtract the exact rounded contribution from the residual.
        residual = residual - contrib

    # Final check
    if np.max(np.abs(residual)) < 1e-12:
        return U, V, W
    return None

def _exact_local_solve(U, V, W, n, m, p, target_rank, max_iters=50, deadline=None):
    """
    Final high-precision polish: alternate exact least-solves for each factor given the
    other two, then verify.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break

        for i in range(n):
            for j in range(m):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for k in range(p):
                    num += V[j*p+k, :] * W[k*n+i, :]
                    den += V[j*p+k, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-12
                if np.any(mask):
                    U[i*m+j, mask] = num[mask] / den[mask]

        for j in range(m):
            for k in range(p):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for i in range(n):
                    num += U[i*m+j, :] * W[k*n+i, :]
                    den += U[i*m+j, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-12
                if np.any(mask):
                    V[j*p+k, mask] = num[mask] / den[mask]

        for k in range(p):
            for i in range(n):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for j in range(m):
                    num += U[i*m+j, :] * V[j*p+k, :]
                    den += U[i*m+j, :]**2 * V[j*p+k, :]**2
                mask = den > 1e-12
                if np.any(mask):
                    W[k*n+i, mask] = num[mask] / den[mask]

        Ur = U.reshape(n, m, target_rank)
        Vr = V.reshape(m, p, target_rank)
        Wr = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
        residual = np.max(np.abs(T - 1.0))
        if residual < 1e-10:
            break

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before the 150s kill.

    # 1. Generate Rank 65 Baseline
    U65, V65, W65 = _construct_rank_65(n, m, p)

    if not _verify_fast(U65, V65, W65, n, m, p):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = np.zeros((n*m, rank_schoolbook))
        V_school = np.zeros((m*p, rank_schoolbook))
        W_school = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    best_U, best_V, best_W = U65, V65, W65
    best_rank = 65

    # 2. Linear algebraic peeling to reduce rank.
    # The baseline is exact, so the residual is zero; peeling must start from a
    # *perturbed* higher-rank representation so that a residual exists to peel.
    # We create a rank-80 schoolbook (exact) and peel it down.
    rank_schoolbook = n * m * p
    U_s = np.zeros((n*m, rank_schoolbook))
    V_s = np.zeros((m*p, rank_schoolbook))
    W_s = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1

    # Peeling the exact schoolbook residual is zero, so instead we peel a
    # *randomized* rank-80 representation that is exact in expectation but has a
    # nonzero residual structure to extract from. To guarantee exactness of the
    # final answer, we peel and then verify; if verification fails we fall back.
    #
    # Strategy: start from the rank-65 baseline, add a small random exact
    # "noise" term (a rank-1 term with a matching negative partner to keep the
    # sum exact) so the residual is nonzero, then peel. Simpler: peel directly
    # from a rank-80 baseline whose columns are randomized but still sum to the
    # identity tensor.
    #
    # Build a randomized exact rank-80: split each schoolbook term into two
    # rank-1 terms with random half-integer scalings that multiply to 1.
    rng = np.random.RandomState(12345)
    U_r80 = np.zeros((n*m, rank_schoolbook))
    V_r80 = np.zeros((m*p, rank_schoolbook))
    W_r80 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Two rank-1 terms whose product contribution is 1:
                # (a * 1)(1/a * 1) with a in {1, 0.5, 2, 1.5, 0.25, ...}
                a = float(rng.choice([1.0, 0.5, 2.0, 1.5, 0.25, 4.0]))
                b = 1.0 / a
                U_r80[i*m+j, col] = a
                V_r80[j*p+k, col] = b
                W_r80[k*n+i, col] = 1.0
                col += 1

    # Verify the randomized rank-80 is exact (it should be, since a*b*1 = 1).
    if _verify_fast(U_r80, V_r80, W_r80, n, m, p):
        peeled = _peel_residual(U_r80, V_r80, W_r80, n, m, p, deadline, max_terms=120)
        if peeled is not None:
            U_p, V_p, W_p = peeled
            U_pr = _round_matrix_to_half_int(U_p)
            V_pr = _round_matrix_to_half_int(V_p)
            W_pr = _round_matrix_to_half_int(W_p)
            if _verify_fast(U_pr, V_pr, W_pr, n, m, p):
                new_rank = len(U_pr[0])
                if new_rank < best_rank:
                    best_U, best_V, best_W = U_pr, V_pr, W_pr
                    best_rank = new_rank

    # 3. Also attempt peeling directly from the rank-65 baseline after adding
    #    an exact rank-2 "wobble" to create a residual.
    if time.time() < deadline and best_rank > 55:
        U_w = U65.copy()
        V_w = V65.copy()
        W_w = W65.copy()
        # Add an exact rank-2 wobble: two rank-1 terms that cancel.
        # term A: uA, vA, wA ; term B: -uA, vA, wA  -> sum zero.
        rng2 = np.random.RandomState(777)
        uA = rng2.randn(n*m)
        vA = rng2.randn(m*p)
        wA = rng2.randn(p*n)
        # Round to half-integers to keep factors in the allowed set.
        uA = np.array([_round_to_half_integer(x) for x in uA])
        vA = np.array([_round_to_half_integer(x) for x in vA])
        wA = np.array([_round_to_half_integer(x) for x in wA])
        # Ensure the wobble is nonzero.
        if np.max(np.abs(uA)) > 1e-12 and np.max(np.abs(vA)) > 1e-12 and np.max(np.abs(wA)) > 1e-12:
            U_w = np.column_stack([U_w, uA, -uA])
            V_w = np.column_stack([V_w, vA, vA])
            W_w = np.column_stack([W_w, wA, wA])
            # Verify the wobbled representation is still exact.
            if _verify_fast(U_w, V_w, W_w, n, m, p):
                peeled = _peel_residual(U_w, V_w, W_w, n, m, p, deadline, max_terms=120)
                if peeled is not None:
                    U_p, V_p, W_p = peeled
                    U_pr = _round_matrix_to_half_int(U_p)
                    V_pr = _round_matrix_to_half_int(V_p)
                    W_pr = _round_matrix_to_half_int(W_p)
                    if _verify_fast(U_pr, V_pr, W_pr, n, m, p):
                        new_rank = len(U_pr[0])
                        if new_rank < best_rank:
                            best_U, best_V, best_W = U_pr, V_pr, W_pr
                            best_rank = new_rank

    # 4. Final high-precision polish on the best found (if below 65).
    if best_rank < 65 and time.time() < deadline:
        U_final = np.array(best_U, dtype=float)
        V_final = np.array(best_V, dtype=float)
        W_final = np.array(best_W, dtype=float)
        U_final, V_final, W_final = _exact_local_solve(
            U_final, V_final, W_final, n, m, p, best_rank,
            max_iters=80, deadline=deadline
        )
        U_r = _round_matrix_to_half_int(U_final)
        V_r = _round_matrix_to_half_int(V_final)
        W_r = _round_matrix_to_half_int(W_final)
        if _verify_fast(U_r, V_r, W_r, n, m, p):
            best_U, best_V, best_W = U_r, V_r, W_r
        # else keep pre-polish

    # Final safety: ensure the returned object is valid; if not, fall back to rank-65.
    if not _verify_fast(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U65, V65, W65

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
