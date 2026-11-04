"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # 2x2x2 Strassen decomposition (rank 7)
    def _strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 0: (A00+A11)*(B00+B11) -> C00 + C11
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: (A10+A11)*B00 -> C10
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # Term 2: A00*(B01-B11) -> C01
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # Term 3: A11*(B10-B00) -> C00 + C10
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 4: (A00+A01)*B11 -> -C00 + C01
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # Term 5: (A10-A00)*(B00+B01) -> C11
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: (A01-A11)*(B10+B11) -> C00 + C11
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    def _verify_2x2x2(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U_arr[i*2+j, c] * V_arr[j*2+k, c] * W_arr[k*2+i, c] for c in range(7))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    U2, V2, W2 = _strassen_2x2x2()
    assert _verify_2x2x2(U2, V2, W2), "2x2x2 Strassen failed"
    
    # 4x4x4 Strassen decomposition (rank 49)
    # 7 block products, each expanded via 2x2x2 Strassen (7 terms each)
    
    # Block index to (bi, bj)
    def block_to_bi_bj(b):
        return b // 2, b % 2
    
    # Global row for A block b, local (li, lj): (2*bi+li)*4 + (2*bj+lj)
    def a_global_row(b, li, lj):
        bi, bj = block_to_bi_bj(b)
        return (2*bi + li) * 4 + (2*bj + lj)
    
    # Global row for B block b, local (lj, lk): (2*bj+lj)*4 + (2*bk+lk)
    def b_global_row(b, lj, lk):
        bj, bk = block_to_bi_bj(b)
        return (2*bj + lj) * 4 + (2*bk + lk)
    
    # Global row for C block b, local (lk, li): (2*bk+lk)*4 + (2*bi+li)
    def c_global_row(b, lk, li):
        bk, bi = block_to_bi_bj(b)
        return (2*bk + lk) * 4 + (2*bi + li)
    
    # 7 Strassen block products for 4x4x4
    # Each: list of (block, sign) for A, B, C
    strassen_4x4 = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
    ]
    
    rank_444 = 49
    U444 = [[0]*rank_444 for _ in range(16)]
    V444 = [[0]*rank_444 for _ in range(16)]
    W444 = [[0]*rank_444 for _ in range(16)]
    
    col = 0
    for prod in strassen_4x4:
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        u_val = U2[li*2+lj][t]
                        v_val = V2[lj*2+lk][t]
                        w_val = W2[lk*2+li][t]
                        
                        for (ab, a_sign) in prod['A']:
                            for (bb, b_sign) in prod['B']:
                                for (cb, c_sign) in prod['C']:
                                    u_row = a_global_row(ab, li, lj)
                                    v_row = b_global_row(bb, lj, lk)
                                    w_row = c_global_row(cb, lk, li)
                                    
                                    U444[u_row][col] += a_sign * u_val
                                    V444[v_row][col] += b_sign * v_val
                                    W444[w_row][col] += c_sign * w_val
            col += 1
    
    assert col == 49, f"Expected 49, got {col}"
    
    # Verify 4x4x4
    def _verify_4x4x4(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = sum(U_arr[i*4+j, c] * V_arr[j*4+k, c] * W_arr[k*4+i, c] for c in range(49))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Extend to 4x4x5: add 1 term for the 5th column
    total_rank = 50
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(20)]
    W_full = [[0]*total_rank for _ in range(20)]
    
    for i in range(16):
        for c in range(49):
            U_full[i][c] = U444[i][c]
    for i in range(16):
        for c in range(49):
            V_full[i][c] = V444[i][c]
    for i in range(16):
        for c in range(49):
            W_full[i][c] = W444[i][c]
    
    # 5th column term
    for i in range(4):
        for j in range(4):
            U_full[i*4+j][49] = 1
    for j in range(4):
        V_full[j*4+4][49] = 1
    for i in range(4):
        W_full[16+i][49] = 1
    
    # Verify full
    def _verify_full(U, V, W, n, m, p, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    assert _verify_full(U_full, V_full, W_full, n, m, p, total_rank), "Full verification failed"
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
