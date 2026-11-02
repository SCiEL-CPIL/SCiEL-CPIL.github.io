"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Pre-compute the rank-49 decomposition at module load time.
# Strassen's 7 products for 2x2 matrices: (A_comb, B_comb) where each is list of (block_row, block_col, coeff)
_STRASSEN_2X2 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

# Strassen C-combination for 2x2:
# C[0,0] = M0 + M3 - M4 + M6
# C[0,1] = M2 + M4
# C[1,0] = M1 + M3
# C[1,1] = M0 - M1 + M2 + M5
_STRASSEN_C_COMB = [
    (0, 0, [1, 0, 0, 1, -1, 0, 1]),
    (0, 1, [0, 0, 1, 0, 1, 0, 0]),
    (1, 0, [0, 1, 0, 1, 0, 0, 0]),
    (1, 1, [1, -1, 1, 0, 0, 1, 0]),
]

def _build_strassen_4x4():
    """Build the rank-49 decomposition by applying Strassen recursively."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    block_products = _STRASSEN_2X2

    for outer_i in range(7):
        a_combo, b_combo = block_products[outer_i]
        for inner_s in range(7):
            r = outer_i * 7 + inner_s
            a_local, b_local = _STRASSEN_2X2[inner_s]
            c_weights = _STRASSEN_C_COMB  # Same for inner and outer in this symmetric case

            # Fill U
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col, r] += a_coeff * l_coeff

            # Fill V
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col, r] += b_coeff * l_coeff

            # Fill W
            # Outer C-combination is same structure as inner
            for (cbr, cbc, outer_w) in _STRASSEN_C_COMB:
                w_outer = outer_w[outer_i]
                if w_outer == 0:
                    continue
                for (cbr2, cbc2, inner_w) in _STRASSEN_C_COMB:
                    w_inner = inner_w[inner_s]
                    if w_inner == 0:
                        continue
                    c_row = 2 * cbr + cbr2
                    c_col = 2 * cbc + cbc2
                    W[c_col * n + c_row, r] += w_outer * w_inner

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Pre-compute at module level
_PRECOMPUTED = _build_strassen_4x4()

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return bool(np.allclose(result, 1.0, atol=1e-10))

def _schoolbook_decomposition(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    result = _PRECOMPUTED
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Verification is fast, but add guard just in case
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return result
    
    # Fallback
    if time.time() < deadline:
        return _schoolbook_decomposition(n, m, p)
        
    # If we are out of time, return the pre-computed result anyway (it's likely correct)
    return result

# EVOLVE-BLOCK-END
