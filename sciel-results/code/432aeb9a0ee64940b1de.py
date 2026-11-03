import time
import numpy as np

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6
NM = N * M  # 8
MP = M * P  # 24
PN = P * N  # 12

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 130.0
    
    rank = 42
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # Strassen coefficients for 2x2 by 2x2
    # u[i][j], v[j][k], w[k][i]
    # i,j,k in {0,1}
    # Term 1: M1 = (a+d)(e+h)
    term1_u = { (0,0): 1, (1,1): 1 }
    term1_v = { (0,0): 1, (1,1): 1 }
    term1_w = { (0,0): 1, (1,1): 1 }
    
    # Term 2: M2 = (c+d)e
    term2_u = { (1,0): 1, (1,1): 1 }
    term2_v = { (0,0): 1 }
    term2_w = { (0,1): 1, (1,1): -1 }
    
    # Term 3: M3 = a(f-h)
    term3_u = { (0,0): 1 }
    term3_v = { (0,1): 1, (1,1): -1 }
    term3_w = { (1,0): 1, (1,1): 1 }
    
    # Term 4: M4 = d(g-e)
    term4_u = { (1,1): 1 }
    term4_v = { (1,0): 1, (0,0): -1 }
    term4_w = { (0,0): 1, (0,1): 1 }
    
    # Term 5: M5 = (a+b)h
    term5_u = { (0,0): 1, (0,1): 1 }
    term5_v = { (1,1): 1 }
    term5_w = { (0,0): -1, (1,0): 1 }
    
    # Term 6: M6 = (c-a)(e+f)
    term6_u = { (1,0): 1, (0,0): -1 }
    term6_v = { (0,0): 1, (0,1): 1 }
    term6_w = { (1,1): 1 }
    
    # Term 7: M7 = (b-d)(g+h)
    term7_u = { (0,1): 1, (1,1): -1 }
    term7_v = { (1,0): 1, (1,1): 1 }
    term7_w = { (0,0): 1 }
    
    terms = [
        (term1_u, term1_v, term1_w),
        (term2_u, term2_v, term2_w),
        (term3_u, term3_v, term3_w),
        (term4_u, term4_v, term4_w),
        (term5_u, term5_v, term5_w),
        (term6_u, term6_v, term6_w),
        (term7_u, term7_v, term7_w),
    ]
    
    col = 0
    # Blocks:
    # 1. A_L (cols 0,1) x B_T (rows 0,1, cols 0,1) -> C (rows 0,1, cols 0,1)
    # 2. A_L (cols 0,1) x B_T (rows 0,1, cols 2,3) -> C (rows 0,1, cols 2,3)
    # 3. A_L (cols 0,1) x B_T (rows 0,1, cols 4,5) -> C (rows 0,1, cols 4,5)
    # 4. A_R (cols 2,3) x B_B (rows 2,3, cols 0,1) -> C (rows 0,1, cols 0,1)
    # 5. A_R (cols 2,3) x B_B (rows 2,3, cols 2,3) -> C (rows 0,1, cols 2,3)
    # 6. A_R (cols 2,3) x B_B (rows 2,3, cols 4,5) -> C (rows 0,1, cols 4,5)
    
    blocks = [
        (0, 0, 0, 0, 0), # a_row, a_col, b_row, b_col, c_col
        (0, 0, 0, 2, 2),
        (0, 0, 0, 4, 4),
        (0, 2, 2, 0, 0),
        (0, 2, 2, 2, 2),
        (0, 2, 2, 4, 4),
    ]
    
    for a_row, a_col, b_row, b_col, c_col in blocks:
        for u_dict, v_dict, w_dict in terms:
            if col >= rank:
                break
                
            for (i, j), val in u_dict.items():
                u_idx = (a_row + i) * M + (a_col + j)
                U[u_idx][col] += val
                
            for (j, k), val in v_dict.items():
                v_idx = (b_row + j) * P + (b_col + k)
                V[v_idx][col] += val
                
            for (k, i), val in w_dict.items():
                w_idx = (c_col + k) * N + (a_row + i)
                W[w_idx][col] += val
                
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
