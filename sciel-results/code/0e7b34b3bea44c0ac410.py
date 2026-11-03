"""Tensor decomposition for matrix multiplication 3x3 by 3x4."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition (rank 36)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    # Candidate rank 35 decomposition
    # 7 Strassen terms for 2x2x2 (indices 0,1)
    # 19 Schoolbook terms for 3x3x3 involving index 2
    # 9 Schoolbook terms for 3x3x1 (index 3)
    rank_cand = 35
    U_cand = [[0] * rank_cand for _ in range(n * m)]
    V_cand = [[0] * rank_cand for _ in range(m * p)]
    W_cand = [[0] * rank_cand for _ in range(p * n)]
    
    # Strassen terms for i,j,k in {0,1}
    # P1: (A00+A11)(B00+B11) -> C00, C11
    # P2: (A10+A11)B00 -> C10, -C11
    # P3: A00(B10-B00) -> C01
    # P4: A11(B11-B01) -> C00, C10
    # P5: (A00+A01)B11 -> -C00, C01
    # P6: (A10-A00)(B00+B01) -> C11
    # P7: (A01-A11)(B10+B11) -> -C00, C11
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: (i,j) -> coeff, V: (j,k) -> coeff, W: (k,i) -> coeff
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # P1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}), # P2 (C10=1, C11=-1)
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1}), # P3 (C01=1)
        ({(1,1): 1}, {(1,1): 1, (0,1): -1}, {(0,0): 1, (0,1): 1}), # P4 (C00=1, C10=1)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}), # P5 (C00=-1, C01=1)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # P6 (C11=1)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,1): 1}) # P7 (C00=-1, C11=1)
    ]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
        for (i, j), coeff in u_dict.items():
            U_cand[i * m + j][r] += coeff
        for (j, k), coeff in v_dict.items():
            V_cand[j * p + k][r] += coeff
        for (k, i), coeff in w_dict.items():
            W_cand[k * n + i][r] += coeff
            
    # Schoolbook terms for 3x3x3 involving index 2
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(3):
                if i == 2 or j == 2 or k == 2:
                    U_cand[i * m + j][col] = 1
                    V_cand[j * p + k][col] = 1
                    W_cand[k * n + i][col] = 1
                    col += 1
                    
    # Schoolbook terms for 3x3x1 (k=3)
    for i in range(n):
        for j in range(m):
            U_cand[i * m + j][col] = 1
            V_cand[j * p + 3][col] = 1
            W_cand[3 * n + i][col] = 1
            col += 1
            
    # Verify candidate
    try:
        U_np = np.array(U_cand, dtype=float)
        V_np = np.array(V_cand, dtype=float)
        W_np = np.array(W_cand, dtype=float)
        
        # Check tensor identity
        # T_ijk = sum_r U_ij,r V_jk,r W_ki,r
        # We want T_ijk = 1 for all i,j,k in 3x3x4
        
        # Construct full tensor
        # U is 9x35, V is 12x35, W is 12x35
        # We need to check for each i,j,k
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i*m+j, :] dot V[j*p+k, :] * W[k*n+i, :]? No.
                    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {"U": U_cand, "V": V_cand, "W": W_cand}
    except Exception:
        pass
        
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
