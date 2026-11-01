import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 23
    rank = 23
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U, V, W entries for each Strassen term
    # Indexing: U[i, j], V[j, k], W[k, i]
    # i, j, k in {0, 1}
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U is 2x2, V is 2x2, W is 2x2
        # We store them as flat lists or dicts
    ]
    
    # Let's define the 7 terms explicitly
    # Term 0: M1
    u0 = { (0,0): 1, (1,1): 1 }
    v0 = { (0,0): 1, (1,1): 1 }
    w0 = { (0,0): 1, (0,1): 1, (1,0): 1, (1,1): -1 } # c00=M1+M4-M5+M7, c01=M3+M5, c10=M2+M4, c11=M1-M2+M3+M6
    # Wait, W[k, i] is the coefficient for C[i, k]?
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Decomposition: C[i, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] ??
    # The task says: W[k*n+i, r]. So W is indexed by [k, i].
    # The term is U[i, j, r] * V[j, k, r] * W[k, i, r].
    # So for a given r, we have a tensor T_r[i, j, k] = U[i, j, r] V[j, k, r] W[k, i, r].
    # We need sum_r T_r[i, j, k] = T[i, j, k].
    
    # For Strassen, the output C is computed as linear combinations of M's.
    # M_r = sum_{i, j} u_{i, j, r} a_{i, j} * sum_{j, k} v_{j, k, r} b_{j, k} ?
    # No, M_r is a scalar product of two vectors.
    # M_r = (sum u_{i, j, r} a_{i, j}) * (sum v_{j, k, r} b_{j, k})
    # Then C_{i, k} = sum_r w_{k, i, r} M_r.
    # So T_r[i, j, k] = u_{i, j, r} v_{j, k, r} w_{k, i, r}.
    
    # Let's extract u, v, w for Strassen terms.
    
    terms = []
    
    # M1
    u = [[1, 0], [0, 1]]
    v = [[1, 1], [1, 1]] # b00+b11 for both? No.
    # M1 = (a00+a11)(b00+b11)
    # u1 = [[1, 0], [0, 1]]
    # v1 = [[1, 1], [1, 1]]? No, v is 2x2.
    # b00+b11 is (1, 0) dot (b00, b01) + (0, 1) dot (b10, b11)?
    # No, b00+b11 is a scalar.
    # v_{j, k} such that sum v_{j, k} b_{j, k} = b00 + b11.
    # v = [[1, 0], [0, 1]].
    # w1: c00 += 1, c11 += 1.
    # w = [[1, 0], [0, 1]].
    terms.append(([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    
    # M2 = (a10+a11)b00
    # u2 = [[0, 0], [1, 1]]
    # v2 = [[1, 0], [0, 0]]
    # w2: c10 += 1.
    # w = [[0, 0], [1, 0]]
    terms.append(([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]))
    
    # M3 = a00(b01-b11)
    # u3 = [[1, 0], [0, 0]]
    # v3 = [[0, 1], [0, -1]]
    # w3: c01 += 1, c11 += 1.
    # w = [[0, 0], [1, 1]]
    terms.append(([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 0], [1, 1]]))
    
    # M4 = a11(b10-b00)
    # u4 = [[0, 0], [0, 1]]
    # v4 = [[-1, 0], [1, 0]]
    # w4: c00 += 1, c10 += 1.
    # w = [[1, 0], [1, 0]]
    terms.append(([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]))
    
    # M5 = (a00+a01)b11
    # u5 = [[1, 1], [0, 0]]
    # v5 = [[0, 0], [0, 1]]
    # w5: c00 -= 1, c01 += 1.
    # w = [[-1, 0], [1, 0]]
    terms.append(([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [1, 0]]))
    
    # M6 = (a10-a00)(b00+b01)
    # u6 = [[-1, 0], [1, 0]]
    # v6 = [[1, 1], [0, 0]]
    # w6: c11 += 1.
    # w = [[0, 0], [0, 1]]
    terms.append(([[ -1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]))
    
    # M7 = (a01-a11)(b10+b11)
    # u7 = [[0, 1], [0, -1]]
    # v7 = [[0, 0], [1, 1]]
    # w7: c00 += 1.
    # w = [[1, 0], [0, 0]]
    terms.append(([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]))
    
    # Add these 7 terms to blocks 1, 2, 3
    # Block 1: j in 0,1; k in 0,1
    # Block 2: j in 0,1; k in 2,3
    # Block 3: j in 0,1; k in 4,5
    
    for block_idx in range(3):
        k_offset = block_idx * 2
        for r_local in range(7):
            r_global = block_idx * 7 + r_local
            u_mat, v_mat, w_mat = terms[r_local]
            for i in range(2):
                for j in range(2):
                    if u_mat[i][j] != 0:
                        U[i * m + j][r_global] += u_mat[i][j]
            for j in range(2):
                for k in range(2):
                    if v_mat[j][k] != 0:
                        V[j * p + (k_offset + k)][r_global] += v_mat[j][k]
            for k in range(2):
                for i in range(2):
                    if w_mat[k][i] != 0:
                        W[(k_offset + k) * n + i][r_global] += w_mat[k][i]
                        
    # Block 4: j in 0,1; k in 6
    # Rank 1: U=1, V=1, W=1
    r_global = 21
    for i in range(2):
        for j in range(2):
            U[i * m + j][r_global] = 1
    for j in range(2):
        V[j * p + 6][r_global] = 1
    for k in range(1): # k=6
        for i in range(2):
            W[6 * n + i][r_global] = 1
            
    # Block 5: j in 2; k in 0..6
    # Rank 1: U=1, V=1, W=1
    r_global = 22
    for i in range(2):
        U[i * m + 2][r_global] = 1
    for k in range(7):
        V[2 * p + k][r_global] = 1
    for k in range(7):
        for i in range(2):
            W[k * n + i][r_global] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
