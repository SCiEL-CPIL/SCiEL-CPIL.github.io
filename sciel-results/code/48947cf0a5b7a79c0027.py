import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Strassen factors for 2x2x2 matrix multiplication
    # The 7 intermediate products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4,         M1-M2+M3+M6]]
    
    strassen_factors = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        # A is 2x4, B is 4x4, C is 2x4
        # We need to decompose this using 4 Strassen 2x2x2 multiplications
        
        # Sub-product definitions:
        # 0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # 1: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        # 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        # 3: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx, col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx, col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx, col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Convert to lists of lists as required by interface
    U_list = [list(row) for row in U]
    V_list = [list(row) for row in V]
    W_list = [list(row) for row in W]
    
    # Verify the decomposition
    def _verify(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U + V + W):
            return False
        
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U_np[i * m + k, r] * V_np[k * p + j, r] * W_np[j * n + i, r]
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col_fb = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_fb[i * m + k][col_fb] = 1
                V_fb[k * p + j][col_fb] = 1
                W_fb[j * n + i][col_fb] = 1
                col_fb += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
