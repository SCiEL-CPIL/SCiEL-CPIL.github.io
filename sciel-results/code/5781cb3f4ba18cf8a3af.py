import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2
    # m: 1..7
    # u[i,j], v[j,k], w[k,i]
    # i,j,k in {0,1}
    
    strassen_u = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    strassen_v = {
        1: {(0,0): 1, (1,1): -1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    strassen_w = {
        1: {(0,0): 1, (1,1): 1, (1,0): -1, (0,1): -1}, # C00, C11, C01(k=1,i=0), C10(k=0,i=1)
        2: {(1,0): 1, (1,1): 1}, # C10, C11
        3: {(1,0): 1, (0,1): 1}, # C10? No.
        # Let's re-verify Strassen W
        # C00 = M1 + M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 + M2 + M3 + M6
        # w[k,i] is coeff for C[i,k]? No, W index is k*n+i.
        # So W[k,i] corresponds to C[i,k].
        
        1: {(0,0): 1, (1,1): 1, (1,0): -1, (0,1): -1}, # C00, C11, C01, C10
        2: {(0,1): 1, (1,1): 1}, # C10 (k=0,i=1)? No.
        # C10 is i=1, k=0. Index (0,1).
        # C11 is i=1, k=1. Index (1,1).
        # So M2 contributes to C10 and C11.
        2: {(0,1): 1, (1,1): 1},
        3: {(1,0): 1, (1,1): 1}, # C01 (k=1,i=0) and C11 (k=1,i=1)
        4: {(0,1): 1, (0,0): 1}, # C10 (k=0,i=1) and C00 (k=0,i=0)
        5: {(0,0): 1, (1,0): 1}, # C00 and C01
        6: {(1,1): 1}, # C11
        7: {(0,0): 1, (1,0): -1}  # C00 and C01?
        # C00 = M1 + M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 + M2 + M3 + M6
        # Check M7: C00 += M7, C01 -= M7?
        # Standard: C01 = M3 + M5. No M7.
        # C00 = M1 + M4 + M5 + M7.
        # So M7 is +1 for C00.
        # Is M7 in C01? No.
        # So w7: {(0,0): 1}.
    }
    
    # Correct Strassen W
    strassen_w = {
        1: {(0,0): 1, (1,1): 1, (1,0): -1, (0,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(1,0): 1, (1,1): 1},
        4: {(0,1): 1, (0,0): 1},
        5: {(0,0): 1, (1,0): 1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    r_idx = 0
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        k_map = {0: k0, 1: k1}
        
        # 7 Strassen terms
        for m in range(1, 8):
            for (li, lj), val in strassen_u[m].items():
                U[li * 3 + lj, r_idx] += val
            for (lj, lk), val in strassen_v[m].items():
                V[lj * 8 + k_map[lk], r_idx] += val
            for (lk, li), val in strassen_w[m].items():
                W[k_map[lk] * 2 + li, r_idx] += val
            r_idx += 1
            
        # 4 Remainder terms (j=2)
        # A[i,2] * B[2,k] -> C[i,k]
        for li in range(2):
            for lk in range(2):
                k_global = k_map[lk]
                U[li * 3 + 2, r_idx] = 1
                V[2 * 8 + k_global, r_idx] = 1
                W[k_global * 2 + li, r_idx] = 1
                r_idx += 1
                
    return {'U': U, 'V': V, 'W': W}
