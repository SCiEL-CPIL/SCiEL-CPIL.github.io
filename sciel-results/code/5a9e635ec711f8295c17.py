"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # Part 1: m1=1 (j=0)
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 0), r] = 1
            V[v_idx(0, k), r] = 1
            W[w_idx(k, i), r] = 1
            r += 1
    
    # Part 2: m2=2 (j=1,2)
    # Split B into 4 blocks of 2x2
    # Use Strassen for each 2x2x2 multiplication
    
    # Strassen terms definitions
    # Local A: 0:a=A[0,1], 1:b=A[0,2], 2:c=A[1,1], 3:d=A[1,2]
    # Local B: 0:e=B[1,k0], 1:f=B[1,k1], 2:g=B[2,k0], 3:h=B[2,k1]
    # Local C: 0:C00=C[0,k0], 1:C01=C[0,k1], 2:C10=C[1,k0], 3:C11=C[1,k1]
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    a_local_to_global = {
        0: (0, 1),
        1: (0, 2),
        2: (1, 1),
        3: (1, 2)
    }
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b_local_to_global = {
            0: (1, k0),
            1: (1, k1),
            2: (2, k0),
            3: (2, k1)
        }
        
        c_local_to_global = {
            0: (0, k0),
            1: (0, k1),
            2: (1, k0),
            3: (1, k1)
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_a, coeff in u_coeffs.items():
                gi, gj = a_local_to_global[loc_a]
                U[u_idx(gi, gj), r] += coeff
            
            for loc_b, coeff in v_coeffs.items():
                gj, gk = b_local_to_global[loc_b]
                V[v_idx(gj, gk), r] += coeff
            
            for loc_c, coeff in w_coeffs.items():
                gi, gk = c_local_to_global[loc_c]
                W[w_idx(gk, gi), r] += coeff
            
            r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
