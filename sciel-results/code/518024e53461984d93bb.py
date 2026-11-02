import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    num_blocks = p // 2 # 6 blocks of 2 columns
    strassen_rank = 7
    rank = num_blocks * strassen_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # u: [u00, u01, u10, u11]
    # v: [v00, v01, v10, v11]
    # w: [w00, w10, w01, w11] corresponding to C indices [0, 1, 2, 3] for a 2x2 block
    # Note: W index for block t is 4t, 4t+1, 4t+2, 4t+3 for C[0,0], C[1,0], C[0,1], C[1,1]
    # Wait, let's re-verify the W index mapping in the code.
    # W row index is k*n + i.
    # For block t, k=2t and k=2t+1.
    # C[0,0] -> k=2t, i=0 -> 4t
    # C[1,0] -> k=2t, i=1 -> 4t+1
    # C[0,1] -> k=2t+1, i=0 -> 4t+2
    # C[1,1] -> k=2t+1, i=1 -> 4t+3
    # So w vector is [w_00, w_10, w_01, w_11]
    
    strassen_coeffs = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    for t in range(num_blocks):
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            u, v, w = strassen_coeffs[s]
            
            # Fill U
            U[0][col] = u[0]
            U[1][col] = u[1]
            U[2][col] = u[2]
            U[3][col] = u[3]
            
            # Fill V
            # v[0] is b00, v[1] is b01, v[2] is b10, v[3] is b11
            # V index for b00: j=0, k=2t -> 0*12 + 2t = 2t
            # V index for b01: j=0, k=2t+1 -> 0*12 + 2t+1 = 2t+1
            # V index for b10: j=1, k=2t -> 1*12 + 2t = 12+2t
            # V index for b11: j=1, k=2t+1 -> 1*12 + 2t+1 = 13+2t
            V[2*t][col] = v[0]
            V[2*t+1][col] = v[1]
            V[12+2*t][col] = v[2]
            V[13+2*t][col] = v[3]
            
            # Fill W
            # w[0] is c00, w[1] is c10, w[2] is c01, w[3] is c11
            # W index for c00: k=2t, i=0 -> 2t*2 + 0 = 4t
            # W index for c10: k=2t, i=1 -> 2t*2 + 1 = 4t+1
            # W index for c01: k=2t+1, i=0 -> (2t+1)*2 + 0 = 4t+2
            # W index for c11: k=2t+1, i=1 -> (2t+1)*2 + 1 = 4t+3
            W[4*t][col] = w[0]
            W[4*t+1][col] = w[1]
            W[4*t+2][col] = w[2]
            W[4*t+3][col] = w[3]
            
    return {"U": U, "V": V, "W": W}
