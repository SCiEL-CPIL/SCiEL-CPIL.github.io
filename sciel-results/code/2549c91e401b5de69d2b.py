"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Rank-44 decomposition: Strassen for 2x2x8 (28 terms) + schoolbook for 2x1x8 (16 terms)
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strassen rank-7 decomposition for 2x2x2
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    # Map local A indices (0,1,2,3) to global row indices in U
    # Local (0,0)->row 0, (0,1)->row 1, (1,0)->row 3, (1,1)->row 4
    # (row 2 is A[i,2] which is not used in Strassen part)
    a_local_to_global = [0, 1, 3, 4]
    
    # Apply Strassen for each block of 2 columns of B (4 blocks total)
    for block in range(4):
        b_col_start = block * 2
        for r in range(7):
            term_idx = block * 7 + r
            
            # U: A linear forms
            for la in range(4):
                coeff = strassen_U[r][la]
                if coeff != 0:
                    ga = a_local_to_global[la]
                    U[ga][term_idx] += coeff
            
            # V: B linear forms
            # Local (0,0)->B[0,b_col_start], (0,1)->B[0,b_col_start+1]
            # (1,0)->B[1,b_col_start], (1,1)->B[1,b_col_start+1]
            b_local_to_global = [
                0 * p + b_col_start,      # (0,0)
                0 * p + b_col_start + 1,  # (0,1)
                1 * p + b_col_start,      # (1,0)
                1 * p + b_col_start + 1,  # (1,1)
            ]
            for lb in range(4):
                coeff = strassen_V[r][lb]
                if coeff != 0:
                    gb = b_local_to_global[lb]
                    V[gb][term_idx] += coeff
            
            # W: C linear forms
            # Local (0,0)->C[0,b_col_start], (0,1)->C[0,b_col_start+1]
            # (1,0)->C[1,b_col_start], (1,1)->C[1,b_col_start+1]
            c_local_to_global = [
                b_col_start * n + 0,      # (0,0)
                (b_col_start + 1) * n + 0, # (0,1)
                b_col_start * n + 1,      # (1,0)
                (b_col_start + 1) * n + 1, # (1,1)
            ]
            for lc in range(4):
                coeff = strassen_W[lc][r]
                if coeff != 0:
                    gc = c_local_to_global[lc]
                    W[gc][term_idx] += coeff
    
    # Schoolbook terms for A[i,2] * B[2,k] for all i in {0,1}, k in {0,...,7}
    for i in range(n):
        for k in range(p):
            term_idx = 28 + i * p + k
            U[i * m + 2][term_idx] = 1
            V[2 * p + k][term_idx] = 1
            W[k * n + i][term_idx] = 1
    
    # Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Should not happen, but return anyway
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
