import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 21 (3 blocks of Strassen rank 7)
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block b: 0*p+2b, 0*p+2b+1, 1*p+2b, 1*p+2b+1
    # W indices for block b: (2b)*n+0, (2b)*n+1, (2b+1)*n+0, (2b+1)*n+1
    
    # Strassen steps:
    # 1: (a+d)(e+h) -> c00, c11
    # 2: (c+d)e -> c10, c11
    # 3: a(f-h) -> c01, c11
    # 4: d(g-e) -> c00, c10
    # 5: (a+b)h -> c00, c01
    # 6: (c-a)(e+f) -> c11
    # 7: (b-d)(g+h) -> c00
    
    # Define coefficients for U, V, W for each Strassen step
    # U is same for all blocks
    # V and W depend on block index b
    
    strassen_defs = [
        # Step 0
        {
            'u': {0: 1, 3: 1},
            'v': {0: 1, 3: 1}, # v indices local: 0:e, 1:f, 2:g, 3:h
            'w': {(0, 0): 1, (1, 1): 1} # w indices local: (i, k_local)
        },
        # Step 1
        {
            'u': {2: 1, 3: 1},
            'v': {0: 1},
            'w': {(1, 0): 1, (1, 1): -1}
        },
        # Step 2
        {
            'u': {0: 1},
            'v': {1: 1, 3: -1},
            'w': {(0, 1): 1, (1, 1): 1}
        },
        # Step 3
        {
            'u': {3: 1},
            'v': {2: 1, 0: -1},
            'w': {(0, 0): 1, (1, 0): 1}
        },
        # Step 4
        {
            'u': {0: 1, 1: 1},
            'v': {3: 1},
            'w': {(0, 0): -1, (0, 1): 1}
        },
        # Step 5
        {
            'u': {2: 1, 0: -1},
            'v': {0: 1, 1: 1},
            'w': {(1, 1): 1}
        },
        # Step 6
        {
            'u': {1: 1, 3: -1},
            'v': {2: 1, 3: 1},
            'w': {(0, 0): 1}
        }
    ]
    
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            defs = strassen_defs[t]
            
            # Fill U
            for idx, val in defs['u'].items():
                U[idx][r] = val
                
            # Fill V
            # Local V indices: 0:(0,2b), 1:(0,2b+1), 2:(1,2b), 3:(1,2b+1)
            for l_idx, val in defs['v'].items():
                if l_idx == 0:
                    v_global = 0 * p + 2 * b
                elif l_idx == 1:
                    v_global = 0 * p + 2 * b + 1
                elif l_idx == 2:
                    v_global = 1 * p + 2 * b
                elif l_idx == 3:
                    v_global = 1 * p + 2 * b + 1
                V[v_global][r] = val
                
            # Fill W
            # Local W indices: (i, k_local) where k_local is 0 or 1
            for (i, k_local), val in defs['w'].items():
                k_global = 2 * b + k_local
                w_global = k_global * n + i
                W[w_global][r] = val
                
    return {"U": U, "V": V, "W": W}
