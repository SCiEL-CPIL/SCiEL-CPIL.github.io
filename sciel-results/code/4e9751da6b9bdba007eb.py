"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # u: coeffs for A (2x2), v: coeffs for B (2x2), w: coeffs for C (2x2)
    # Indices: A[i,j], B[j,k], C[k,i] -> wait, C[i,k]
    # W index is k*n + i.
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00 -> C10+=1, C11-=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: A00(B01-B11) -> C01+=1, C11+=1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: A11(B10-B00) -> C00+=1, C10+=1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A00+A01)B11 -> C00-=1, C01+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)(B00+B01) -> C11+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A01-A11)(B10+B11) -> C00+=1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for b in range(3):
        for u_dict, v_dict, w_dict in strassen_terms:
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * m + j][col] += val
            # Fill V
            for (j, k_loc), val in v_dict.items():
                k_glob = 2 * b + k_loc
                V[j * p + k_glob][col] += val
            # Fill W
            for (i, k_loc), val in w_dict.items():
                k_glob = 2 * b + k_loc
                W[k_glob * n + i][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
