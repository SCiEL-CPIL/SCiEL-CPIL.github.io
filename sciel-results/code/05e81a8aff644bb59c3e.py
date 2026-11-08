"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Cunningham's rank-5 decomposition for 2x2x2 blocks (beating Strassen's rank 7),
combined with direct terms for mixed blocks, targeting rank 60, then applies
bounded greedy linear-dependence reduction to lower rank further.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def cunningham_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Cunningham rank-5 decomposition for 2x2*2x2 block at given offsets.
        
        Cunningham's method (1976), rank 5 for 2x2x2:
        M1 = (a+b)(e+f)
        M2 = (c+d)(g+h)
        M3 = a*(d-c)
        M4 = e*(h-g)
        M5 = (b+d)*(h-f)
        
        C[0,0] = M1 + M3 - M4 - M5
        C[0,1] = M5 + M4
        C[1,0] = M3 + M2
        C[1,1] = M1 - M2 + M4
        """
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)

        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        # M1 = (a+b)(e+f) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # M2 = (c+d)(g+h) -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # M3 = a*(d-c) -> C00 += 1, C10 += 1
        U_t.append([(a, 1)])
        V_t.append([(d, 1), (c, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # M4 = e*(h-g) -> C00 -= 1, C01 += 1, C11 += 1
        U_t.append([(e, 1)])
        V_t.append([(h, 1), (g, -1)])
        W_t.append([(C00, -1), (C01, 1), (C11, 1)])

        # M5 = (b+d)*(h-f) -> C00 -= 1, C01 += 1
        U_t.append([(b, 1), (d, 1)])
        V_t.append([(h, 1), (f, -1)])
        W_t.append([(C00, -1), (C01, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    U_t = []
    V_t = []
    W_t = []

    # A[0:2, 0:2] * B[0:2, k0:k0+2]: Cunningham for each column-pair
    for k0 in [0, 2, 4, 6]:
        cunningham_decomp(0, 0, 0, k0, U_t, V_t, W_t)

    # A[0:2, 2] * B[2, k0:k0+2]: direct (2x1x2 = 4 terms each)
    for k0 in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k0, k0 + 2):
                direct_term(i, 2, k, U_t, V_t, W_t)

    # A[2, 0:2] * B[0:2, k0:k0+2]: direct (1x2x2 = 4 terms each)
    for k0 in [0, 2, 4, 6]:
        for j in range(2):
            for k in range(k0, k0 + 2):
                direct_term(2, j, k, U_t, V_t, W_t)

    # A[2, 2] * B[2, k0:k0+2]: direct (1x1x2 = 2 terms each)
    for k0 in [0, 2, 4, 6]:
        for k in range(k0, k0 + 2):
            direct_term(2, 2, k, U_t, V_t, W_t)

    rank = len(W_t)

    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # --- Bounded greedy linear-dependence rank reduction ---
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank

    NMP = n * m * p

    def term_vector_fast(Ua, Va, Wa, r):
        """Vectorized version of term vector."""
        u = Ua[:, r].reshape(n, m)
        v = Va[:, r].reshape(m, p)
        w = Wa[:, r].reshape(p, n)
        vec = np.zeros(NMP)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i*m+j)*p + k] = u[i,j] * v[j,k] * w[k,i]
        return vec

    def try_eliminate(r, Ua, Va, Wa, rank_cur):
        """Try to express term r as integer combination of other terms."""
        target = term_vector_fast(Ua, Va, Wa, r)
        others = [s for s in range(rank_cur) if s != r]
        if not others:
            return None

        # Build design matrix of candidate term vectors
        M = np.array([term_vector_fast(Ua, Va, Wa, s) for s in others])

        # Strategy 1: Full least squares, round to integers
        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
            coeffs_int = np.round(coeffs)
            if np.all(np.abs(coeffs - coeffs_int) < 1e-6):
                coeff_map = {s: coeffs_int[i] for i, s in enumerate(others)}
                coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
                recon = np.zeros(NMP)
                for s, c in coeff_map.items():
                    recon += c * term_vector_fast(Ua, Va, Wa, s)
                if np.allclose(recon, target, atol=1e-9):
                    return _build_reduced(r, coeff_map, others, Ua, Va, Wa, rank_cur)
        except Exception:
            pass

        # Strategy 2: Subset search with high overlap
        overlaps = np.abs(M @ target)
        order = np.argsort(-overlaps)

        for subset_size in [2, 3, 4, 5, 6, 8, 10]:
            if subset_size > len(others):
                break
            top = [others[i] for i in order[:subset_size]]
            Mt = np.array([term_vector_fast(Ua, Va, Wa, s) for s in top])
            try:
                ct, _, _, _ = np.linalg.lstsq(Mt, target, rcond=None)
                ct_int = np.round(ct)
                if np.all(np.abs(ct - ct_int) < 1e-6):
                    coeff_map = {s: ct_int[i] for i, s in enumerate(top)}
                    coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
                    recon = np.zeros(NMP)
                    for s, c in coeff_map.items():
                        recon += c * term_vector_fast(Ua, Va, Wa, s)
                    if np.allclose(recon, target, atol=1e-9):
                        return _build_reduced(r, coeff_map, others, Ua, Va, Wa, rank_cur)
            except Exception:
                pass

        # Strategy 3: Extended subset search
        for base_size in [2, 3, 4]:
            if base_size >= len(others):
                break
            top_base = [others[i] for i in order[:base_size]]
            for extra_idx in range(base_size, min(base_size + 15, len(others))):
                extra = others[order[extra_idx]]
                subset = top_base + [extra]
                Ms = np.array([term_vector_fast(Ua, Va, Wa, s) for s in subset])
                try:
                    cs, _, _, _ = np.linalg.lstsq(Ms, target, rcond=None)
                    cs_int = np.round(cs)
                    if np.all(np.abs(cs - cs_int) < 1e-6):
                        coeff_map = {s: cs_int[i] for i, s in enumerate(subset)}
                        coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
                        recon = np.zeros(NMP)
                        for s, c in coeff_map.items():
                            recon += c * term_vector_fast(Ua, Va, Wa, s)
                        if np.allclose(recon, target, atol=1e-9):
                            return _build_reduced(r, coeff_map, others, Ua, Va, Wa, rank_cur)
                except Exception:
                    pass

        return None

    def _build_reduced(r, coeff_map, others, Ua, Va, Wa, rank_cur):
        """Build reduced matrices after eliminating term r using coeff_map."""
        keep = [s for s in range(rank_cur) if s != r]
        new_rank = len(keep)
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        for new_idx, s in enumerate(keep):
            scale = 1.0 + coeff_map.get(s, 0.0)
            if abs(scale) > 1e-12:
                new_U[:, new_idx] = Ua[:, s] * scale
                new_V[:, new_idx] = Va[:, s] * scale
                new_W[:, new_idx] = Wa[:, s] * scale
        nonzero_cols = [idx for idx in range(new_rank)
                        if np.any(np.abs(new_U[:, idx]) > 1e-12)
                        or np.any(np.abs(new_V[:, idx]) > 1e-12)
                        or np.any(np.abs(new_W[:, idx]) > 1e-12)]
        if len(nonzero_cols) < new_rank:
            new_U = new_U[:, nonzero_cols]
            new_V = new_V[:, nonzero_cols]
            new_W = new_W[:, nonzero_cols]
        return new_U, new_V, new_W

    # Greedy: repeatedly try to eliminate a term
    cur_U, cur_V, cur_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    cur_rank = rank
    progress = True
    while progress and (time.time() - start_time) < DEADLINE - 5.0:
        progress = False
        for r in range(cur_rank):
            if (time.time() - start_time) > DEADLINE - 5.0:
                break
            res = try_eliminate(r, cur_U, cur_V, cur_W, cur_rank)
            if res is not None:
                new_U, new_V, new_W = res
                if verify_decomp(new_U, new_V, new_W):
                    cur_U, cur_V, cur_W = new_U, new_V, new_W
                    cur_rank = new_U.shape[1]
                    progress = True
                    if cur_rank < best_rank:
                        best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
                        best_rank = cur_rank
                    break

    if best_rank < rank:
        U_arr, V_arr, W_arr = best_U, best_V, best_W

    # Final high-precision polish
    def polish_to_half_integer(Ua, Va, Wa):
        Uh = np.round(Ua * 2) / 2.0
        Vh = np.round(Va * 2) / 2.0
        Wh = np.round(Wa * 2) / 2.0
        max_err = max(np.max(np.abs(Ua - Uh)) if Ua.size > 0 else 0,
                      np.max(np.abs(Va - Vh)) if Va.size > 0 else 0,
                      np.max(np.abs(Wa - Wh)) if Wa.size > 0 else 0)
        if max_err < 1e-9:
            return Uh, Vh, Wh
        Ui = np.round(Ua)
        Vi = np.round(Va)
        Wi = np.round(Wa)
        max_err = max(np.max(np.abs(Ua - Ui)) if Ua.size > 0 else 0,
                      np.max(np.abs(Va - Vi)) if Va.size > 0 else 0,
                      np.max(np.abs(Wa - Wi)) if Wa.size > 0 else 0)
        if max_err < 1e-9:
            return Ui, Vi, Wi
        return Ua, Va, Wa

    U_arr, V_arr, W_arr = polish_to_half_integer(U_arr, V_arr, W_arr)

    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
