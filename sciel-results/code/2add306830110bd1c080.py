import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen terms for 2x2x2x2 block
    # Each term: (U_sparsity, V_sparsity, W_sparsity)
    # Indices: (row, col) for U, (row, col) for V, (row, col) for W
    # where for a 2x2 block, rows/cols are 0 or 1
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # M2 = (A10+A11)B00 -> contributes to C10
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3 = A00(B10-B00) -> contributes to C00
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,0):1}),
        # M4 = A11(B01-B11) -> contributes to C11
        ({(1,1):1}, {(0,1):1, (1,1):-1}, {(1,1):1}),
        # M5 = (A00+A01)B11 -> contributes to C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        # M6 = (A10-A00)(B00+B01) -> contributes to C10, C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),
        # M7 = (A01-A11)(B10+B11) -> contributes to C00, C01
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}),
    ]
    
    col = 0
    for a in range(2):
        for b in range(4):
            for t in range(7):
                if time.time() > deadline - 10:
                    return {"U": U, "V": V, "W": W}
                
                u_terms, v_terms, w_terms = strassen_terms[t]
                
                for (i_prime, j_prime), coeff in u_terms.items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx][col] += coeff
                
                for (j_prime, k_prime), coeff in v_terms.items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx][col] += coeff
                
                for (k_prime, i_prime), coeff in w_terms.items():
                    # W is indexed as [k*n + i, r]
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
