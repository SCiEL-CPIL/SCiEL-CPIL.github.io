import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # The minimal rank for M_{2,4,8} is 56.
    # We construct the decomposition based on block Strassen.
    # A (2x4) = [A_L A_R] where A_L, A_R are 2x2.
    # B (4x8) = [B_L; B_R] where B_L, B_R are 2x8.
    # C = A_L B_L + A_R B_R.
    # Each term M_{2,2,8} has rank 28.
    # M_{2,2,8} is 4 blocks of M_{2,2,2} (rank 7).
    # Total rank = 2 * 28 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Process A_L B_L
    # A_L is rows 0,1 of A? No, A is 2x4.
    # Let's stick to the indices used in the previous program which worked.
    # Previous program used:
    # n_blocks_i = 1 (i in 0,1)
    # n_blocks_j = 2 (j in 0,1 and 2,3)
    # n_blocks_k = 4 (k in 0,1 ... 6,7)
    # This effectively computes A_L B_L + A_R B_R where B_L is cols 0-3? No.
    
    # Let's just use the logic from the current program which is correct.
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1
                U[u_a, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1; W[w_c11, col] += 1
                col += 1
                # M2
                U[u_c, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1
                W[w_c10, col] += 1; W[w_c11, col] -= 1
                col += 1
                # M3
                U[u_a, col] += 1
                V[v_f, col] += 1; V[v_h, col] -= 1
                W[w_c01, col] += 1; W[w_c11, col] += 1
                col += 1
                # M4
                U[u_d, col] += 1
                V[v_g, col] += 1; V[v_e, col] -= 1
                W[w_c00, col] += 1; W[w_c10, col] += 1
                col += 1
                # M5
                U[u_a, col] += 1; U[u_b, col] += 1
                V[v_h, col] += 1
                W[w_c00, col] -= 1; W[w_c01, col] += 1
                col += 1
                # M6
                U[u_c, col] += 1; U[u_a, col] -= 1
                V[v_e, col] += 1; V[v_f, col] += 1
                W[w_c11, col] += 1
                col += 1
                # M7
                U[u_b, col] += 1; U[u_d, col] -= 1
                V[v_g, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1
                col += 1
                
    # Convert to lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
