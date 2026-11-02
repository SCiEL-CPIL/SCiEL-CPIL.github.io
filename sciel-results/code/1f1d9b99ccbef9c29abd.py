"""Matrix multiplication 3,3,6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify U,V,W exactly decompose the matmul tensor.
    sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _verify_np(U, V, W, n, m, p):
    """Vectorized verification."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
    return np.max(np.abs(T - 1.0)) <= 1e-9


def _round_to_half(x):
    """Round to nearest half-integer; return (value, ok)."""
    d = 2.0 * x
    c = np.round(d)
    ok = np.max(np.abs(d - c)) < 1e-7
    return c / 2.0, ok


def _solve_core(U, V, n, m, p, tol=1e-7):
    """Given U (nm x R) and V (mp x R), solve for W (pn x R) via least squares.
    Returns (W, ok) where W has half-integer entries if ok.
    """
    R = U.shape[1]
    W = np.zeros((n * p, R))
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, R))
            for j in range(m):
                M[j, :] = U[i*m + j] * V[j*p + k]
            try:
                w = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except Exception:
                return None, False
            w, ok = _round_to_half(w)
            if not ok:
                return None, False
            W[i*p + k] = w
    return W, True


def _search_core(n, m, p, R, n_iter, deadline, seed):
    """Search for a rank-R core decomposition. Returns (U,V,W) lists or None."""
    rng = np.random.RandomState(seed)
    for it in range(n_iter):
        if time.time() > deadline:
            return None
        U = rng.randint(-1, 2, size=(n*m, R)).astype(float)
        V = rng.randint(-1, 2, size=(m*p, R)).astype(float)
        W, ok = _solve_core(U, V, n, m, p)
        if not ok:
            continue
        # verify
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        if np.max(np.abs(T - 1.0)) <= 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
    return None


def _search_direct(n, m, p, R, n_iter, deadline, seed):
    """Direct search for rank-R decomposition of n x m x p. Returns dict or None."""
    rng = np.random.RandomState(seed)
    for it in range(n_iter):
        if time.time() > deadline:
            return None
        U = rng.randint(-1, 2, size=(n*m, R)).astype(float)
        V = rng.randint(-1, 2, size=(m*p, R)).astype(float)
        W = np.zeros((p*n, R))
        ok = True
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    w = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except Exception:
                    ok = False
                    break
                w, ok2 = _round_to_half(w)
                if not ok2:
                    ok = False
                    break
                W[k*n + i] = w
            if not ok:
                break
        if not ok:
            continue
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        if np.max(np.abs(T - 1.0)) <= 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_3x3x6_from_core(U1, V1, W1):
    """Build 3x3x6 decomposition from a 3x3x3 core (rank R). Rank 2R."""
    R = len(U1[0])
    n, m, p = 3, 3, 6
    Rtot = 2 * R
    U = [[0] * Rtot for _ in range(n * m)]
    V = [[0] * Rtot for _ in range(m * p)]
    W = [[0] * Rtot for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for r in range(R):
                U[i*m + j][r] = U1[i*m + j][r]
                U[i*m + j][R + r] = U1[i*m + j][r]
    for j in range(m):
        for k in range(3):
            for r in range(R):
                V[j*6 + k][r] = V1[j*3 + k][r]
                V[j*6 + (k + 3)][R + r] = V1[j*3 + k][r]
    for k in range(3):
        for i in range(n):
            for r in range(R):
                W[k*3 + i][r] = W1[k*3 + i][r]
                W[(k + 3)*3 + i][R + r] = W1[k*3 + i][r]
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """7-term rank decomposition for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1,0,1,1],[1,0,0,1],[1,0,0,1]))
    terms.append(([0,0,1,1],[1,0,0,0],[0,0,1,0]))
    terms.append(([1,0,0,0],[0,1,0,-1],[0,1,0,0]))
    terms.append(([0,0,0,1],[0,0,1,0],[1,0,0,0]))
    terms.append(([1,1,0,0],[0,0,0,1],[0,0,0,-1]))
    terms.append([[-1,0,1,0],[1,1,0,0],[0,0,0,1]])
    terms.append(([0,1,0,-1],[0,0,1,1],[1,0,0,1]))
    return terms


def _gf2_rank(matrix):
    """Compute rank of a matrix over GF(2)."""
    M = np.array(matrix, dtype=np.uint8) % 2
    if M.size == 0:
        return 0
    rows, cols = M.shape
    rank = 0
    M = M.copy()
    for c in range(cols):
        if rank >= rows:
            break
        pivot = -1
        for r in range(rank, rows):
            if M[r, c] % 2 == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rows):
            if r != rank and M[r, c] % 2 == 1:
                M[r] = (M[r] + M[rank]) % 2
        rank += 1
    return rank


def _gf2_support_analysis(n, m, p, deadline):
    """
    Analyze the matmul tensor over GF(2) to guide rank search.
    Returns a dict with GF(2) rank info and suggested search parameters.
    """
    # Build the matmul tensor as a 3D array over GF(2)
    # T[i, j, k] = 1 if this is a valid multiplication term
    # For 3x3x6: T[i, j, k] where i in [0,3), j in [0,3), k in [0,6)
    # The tensor is 1 everywhere (each term A[i,j]*B[j,k] contributes to C[i,k])
    # But the structure is: for fixed (i,k), the sum over j of U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Over GF(2), we can look at the "slice" matrices
    # For each r, the contribution is a rank-1 tensor: U_r ⊗ V_r ⊗ W_r (with appropriate indexing)
    
    # Key insight: over GF(2), the matmul tensor has a specific structure
    # Let's compute the "flattened" tensor and its GF(2) properties
    
    # For 3x3x6, the tensor T has T[i,j,k] = 1 for all i,j,k (since every product term exists)
    # Wait, no. The tensor for matmul is:
    # T_{(i,j),(j,k),(i,k)} = 1
    # In our indexing: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's think about this differently.
    # The key structural insight over GF(2):
    # If we reduce the decomposition mod 2, we get a decomposition over GF(2)
    # The GF(2) rank gives a lower bound
    
    # For 3x3x3, the known GF(2) rank is 19 (same as over reals for this case)
    # For 3x3x6, it's 2 * 19 = 38
    
    # Let's verify by computing the tensor and checking properties
    n, m, p = 3, 3, 6
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor T as a 3D array: T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[a, b, c] = 1 if the indices are consistent (j from a matches j from b, 
    #             i from a matches i from c, k from b matches k from c)
    
    T = np.zeros((nm, mp, pn), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Compute GF(2) rank of different flattenings
    # Flattening 1: (nm) x (mp*pn)
    T1 = T.reshape(nm, mp * pn)
    r1 = _gf2_rank(T1)
    
    # Flattening 2: (mp) x (nm*pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    r2 = _gf2_rank(T2)
    
    # Flattening 3: (pn) x (nm*mp)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    r3 = _gf2_rank(T3)
    
    # The tensor rank over GF(2) is at least max(r1, r2, r3)
    gf2_lower_bound = max(r1, r2, r3)
    
    return {
        "gf2_ranks": (r1, r2, r3),
        "gf2_lower_bound": gf2_lower_bound,
        "suggested_start_rank": max(gf2_lower_bound, 30)
    }


def _try_small_rank_decomposition(n, m, p, deadline):
    """Try to find a decomposition with rank < n*m*p.
    Uses GF(2) analysis to guide search, then tries core 3x3x3 (then 3x3x6 = 2*core),
    then direct 3x3x6.
    Returns best dict found or None.
    """
    best = None
    
    # --- GF(2) analysis to guide search ---
    gf2_info = _gf2_support_analysis(n, m, p, deadline)
    gf2_lb = gf2_info["gf2_lower_bound"]
    
    # --- Core 3x3x3 search (gives 3x3x6 rank 2*R) ---
    if (n, m, p) == (3, 3, 6):
        # Start from GF(2) lower bound / 2 for core, but at least 19
        core_start = max(19, gf2_lb // 2)
        for R in range(core_start, 28):
            if time.time() > deadline:
                break
            n_iter = 4000 if R >= 26 else 2000
            found = _search_core(3, 3, 3, R, n_iter, deadline, seed=1000 + R)
            if found is not None:
                U1, V1, W1 = found
                res = _build_3x3x6_from_core(U1, V1, W1)
                if _verify_np(res["U"], res["V"], res["W"], 3, 3, 6):
                    best = res
                    break  # lowest R found, stop

    # --- Direct 3x3x6 search for even lower rank ---
    if time.time() < deadline:
        direct_start = max(30, gf2_lb)
        for R in range(direct_start, 54):
            if time.time() > deadline:
                break
            n_iter = 1500
            res = _search_direct(n, m, p, R, n_iter, deadline, seed=5000 + R)
            if res is not None and _verify_np(res["U"], res["V"], res["W"], n, m, p):
                if best is None or R < len(best["U"][0]):
                    best = res
                break

    return best


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x6 matmul."""
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin for 120s/150s limit
    n, m, p = DIMENSIONS

    # Try to find a low-rank decomposition
    result = _try_small_rank_decomposition(n, m, p, deadline)

    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
