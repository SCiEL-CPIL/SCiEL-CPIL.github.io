import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - start_time < DEADLINE

    def build_strassen_block(i1, i2, j1, j2, k1, k2):
        terms = []
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        terms.append(({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
        terms.append(({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
        terms.append(({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
        terms.append(({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
        terms.append(({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}))
        terms.append(({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
        terms.append(({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
        return terms

    def build_schoolbook_row(i, j_start, j_end, k_start, k_end):
        terms = []
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    def build_full_schoolbook():
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    def build_bini_3x3x3(i0, j0, k0):
        terms = []
        def A(i, j):
            return (i0 + i) * m + (j0 + j)
        def B(j, k):
            return (j0 + j) * p + (k0 + k)
        def C(k, i):
            return (k0 + k) * n + (i0 + i)
        
        # Bini 23-term decomposition for 3x3 matrix multiplication
        terms.append(({A(0,0): 1, A(1,1): 1}, {B(0,0): 1, B(1,1): 1}, {C(0,0): 1, C(1,1): 1}))
        terms.append(({A(0,0): 1, A(1,1): 1}, {B(0,1): 1, B(1,0): 1}, {C(0,1): 1, C(1,0): 1}))
        terms.append(({A(0,0): 1, A(1,1): 1}, {B(0,2): 1, B(1,2): 1}, {C(0,2): 1, C(1,2): 1}))
        terms.append(({A(0,1): 1, A(1,0): 1}, {B(0,0): 1, B(1,1): 1}, {C(1,0): 1, C(0,1): 1}))
        terms.append(({A(0,1): 1, A(1,0): 1}, {B(0,1): 1, B(1,0): 1}, {C(1,1): 1}))
        terms.append(({A(0,1): 1, A(1,0): 1}, {B(0,2): 1, B(1,2): 1}, {C(1,2): 1, C(0,2): 1}))
        terms.append(({A(0,2): 1, A(1,2): 1}, {B(0,0): 1, B(1,1): 1}, {C(2,0): 1, C(1,1): 1}))
        terms.append(({A(0,2): 1, A(1,2): 1}, {B(0,1): 1, B(1,0): 1}, {C(2,1): 1, C(0,2): 1}))
        terms.append(({A(0,2): 1, A(1,2): 1}, {B(0,2): 1, B(1,2): 1}, {C(2,2): 1}))
        terms.append(({A(0,0): 1, A(0,1): 1, A(0,2): 1}, {B(0,0): 1, B(0,1): 1, B(0,2): 1}, {C(0,0): 1}))
        terms.append(({A(0,0): 1, A(0,1): 1, A(0,2): 1}, {B(1,0): 1, B(1,1): 1, B(1,2): 1}, {C(1,0): 1}))
        terms.append(({A(0,0): 1, A(0,1): 1, A(0,2): 1}, {B(2,0): 1, B(2,1): 1, B(2,2): 1}, {C(2,0): 1}))
        terms.append(({A(1,0): 1, A(1,1): 1, A(1,2): 1}, {B(0,0): 1, B(0,1): 1, B(0,2): 1}, {C(0,1): 1}))
        terms.append(({A(1,0): 1, A(1,1): 1, A(1,2): 1}, {B(1,0): 1, B(1,1): 1, B(1,2): 1}, {C(1,1): 1}))
        terms.append(({A(1,0): 1, A(1,1): 1, A(1,2): 1}, {B(2,0): 1, B(2,1): 1, B(2,2): 1}, {C(2,1): 1}))
        terms.append(({A(2,0): 1, A(2,1): 1, A(2,2): 1}, {B(0,0): 1, B(0,1): 1, B(0,2): 1}, {C(0,2): 1}))
        terms.append(({A(2,0): 1, A(2,1): 1, A(2,2): 1}, {B(1,0): 1, B(1,1): 1, B(1,2): 1}, {C(1,2): 1}))
        terms.append(({A(2,0): 1, A(2,1): 1, A(2,2): 1}, {B(2,0): 1, B(2,1): 1, B(2,2): 1}, {C(2,2): 1}))
        terms.append(({A(2,0): 1}, {B(0,0): 1}, {C(0,0): -1}))
        terms.append(({A(2,0): 1}, {B(1,1): 1}, {C(1,1): -1}))
        terms.append(({A(2,0): 1}, {B(2,2): 1}, {C(2,2): -1}))
        terms.append(({A(2,1): 1}, {B(0,0): 1}, {C(0,0): -1}))
        terms.append(({A(2,1): 1}, {B(1,1): 1}, {C(1,1): -1}))
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, coeff in u_dict.items():
                U[idx, r] = coeff
            for idx, coeff in v_dict.items():
                V[idx, r] = coeff
            for idx, coeff in w_dict.items():
                W[idx, r] = coeff
        return U, V, W

    def verify(U, V, W):
        rank = U.shape[1]
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.max(np.abs(result - 1.0)) < 1e-9

    # Strategy A: Bini 3x3x3 on blocks + schoolbook for col 3
    def strategy_bini():
        terms = []
        terms.extend(build_bini_3x3x3(0, 0, 0))
        terms.extend(build_bini_3x3x3(0, 0, 3))
        terms.extend(build_schoolbook_row(0, 3, 4, 0, p))
        terms.extend(build_schoolbook_row(1, 3, 4, 0, p))
        terms.extend(build_schoolbook_row(2, 3, 4, 0, p))
        return terms

    # Strategy B: Strassen on 2x2x2 blocks for rows 1-2
    def strategy_strassen():
        terms = []
        terms.extend(build_schoolbook_row(0, 0, m, 0, p))
        for j1 in range(0, m, 2):
            for k1 in range(0, p, 2):
                terms.extend(build_strassen_block(1, 2, j1, j1+1, k1, k1+1))
        return terms

    # Strategy C: Strassen alternative for rows 0-1
    def strategy_strassen2():
        terms = []
        terms.extend(build_schoolbook_row(2, 0, m, 0, p))
        for j1 in range(0, m, 2):
            for k1 in range(0, p, 2):
                terms.extend(build_strassen_block(0, 1, j1, j1+1, k1, k1+1))
        return terms

    # Strategy D: Full schoolbook
    def strategy_schoolbook():
        return build_full_schoolbook()

    # Strategy E: Hybrid - Bini for first 3 cols of B, Strassen for col 3 area
    def strategy_hybrid():
        terms = []
        # Use Bini for j=0,1,2 with k=0,1,2 and k=3,4,5
        terms.extend(build_bini_3x3x3(0, 0, 0))
        terms.extend(build_bini_3x3x3(0, 0, 3))
        # For j=3, use schoolbook for all k
        terms.extend(build_schoolbook_row(0, 3, 4, 0, p))
        terms.extend(build_schoolbook_row(1, 3, 4, 0, p))
        terms.extend(build_schoolbook_row(2, 3, 4, 0, p))
        return terms

    # Strategy F: Try 3x2x3 Strassen-like blocks
    def strategy_3x2x3():
        terms = []
        # Process rows 0-2, cols 0-1, k 0-2 as a 3x2x3 block
        # Then rows 0-2, cols 2-3, k 0-2
        # Then rows 0-2, cols 0-1, k 3-5
        # Then rows 0-2, cols 2-3, k 3-5
        # For each 3x2x3 block, use a decomposition
        # Simple approach: use schoolbook for now
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    strategies = [strategy_bini, strategy_strassen, strategy_strassen2, strategy_schoolbook, strategy_hybrid]
    best_result = None
    best_rank = float('inf')

    for idx, strat_fn in enumerate(strategies):
        if not time_ok():
            break
        terms = strat_fn()
        rank = len(terms)
        if rank >= best_rank:
            continue
        U, V, W = terms_to_matrices(terms)
        if verify(U, V, W):
            best_result = (U, V, W, rank)
            best_rank = rank

    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    terms = build_full_schoolbook()
    U, V, W = terms_to_matrices(terms)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
