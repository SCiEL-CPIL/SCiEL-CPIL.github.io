import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33
    rank = 33
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: Strassen for 3 blocks of 2x2x2
    # A is 2x2 (cols 0,1), B is 2x2 (cols 2b, 2b+1)
    # i in 0,1; j in 0,1; k in 0,1 (local)
    
    # Define Strassen terms for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    strassen_terms = [
        # (U_vals, V_vals, W_vals)
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): -1}),
    ]
    
    for b in range(3):
        for u_dict, v_dict, w_dict in strassen_terms:
            for (i, j), val in u_dict.items():
                U[i * m + j, col] = val
            for (j, k_local), val in v_dict.items():
                k_global = 2 * b + k_local
                V[j * p + k_global, col] = val
            for (k_local, i), val in w_dict.items():
                k_global = 2 * b + k_local
                W[k_global * n + i, col] = val
            col += 1
            
    # Part 2: Outer product for remaining column of A
    # A is 2x1 (col 2), B is 1x6 (row 2)
    # i in 0,1; j=2; k in 0..5
    for i in range(2):
        for k in range(6):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
