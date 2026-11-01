import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (16, rank)
    
    rank = 29
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Term 0: j=0 part
    # A_{i,0} B_{0,k}
    # U index: 3*i + 0
    # V index: 8*0 + k = k
    # W index: 2*k + i
    
    for i in range(n):
        U[i*m + 0, 0] = 1.0
    for k in range(p):
        V[0*p + k, 0] = 1.0
    for k in range(p):
        for i in range(n):
            W[k*n + i, 0] = 1.0
            
    # Terms 1-28: Strassen for 4 blocks
    # Strassen factors for 2x2
    # u, v, w are vectors of length 4 (indices 0..3 corresponding to 00, 01, 10, 11)
    strassen_u = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 1, -1],# P6
        [0, 1, 0, -1] # P7
    ]
    strassen_v = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    strassen_w = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    rank_idx = 1
    for b in range(4):
        for s in range(7):
            r = rank_idx
            # Process U
            for i_prime in range(2):
                for j_prime in range(2):
                    val = strassen_u[s][i_prime*2 + j_prime]
                    if val != 0:
                        # A'_{i, j'} -> A_{i, j'+1}
                        # U index: 3*i + (j'+1)
                        u_idx = i_prime * m + (j_prime + 1)
                        U[u_idx, r] += val
                        
            # Process V
            for j_prime in range(2):
                for k_prime in range(2):
                    val = strassen_v[s][j_prime*2 + k_prime]
                    if val != 0:
                        # B'_{j', k'} -> B_{j'+1, 2b+k'}
                        # V index: 8*(j'+1) + (2b+k')
                        v_idx = (j_prime + 1) * p + (2*b + k_prime)
                        V[v_idx, r] += val
                        
            # Process W
            for i_prime in range(2):
                for k_prime in range(2):
                    val = strassen_w[s][i_prime*2 + k_prime]
                    if val != 0:
                        # C'_{i, k'} -> C_{i, 2b+k'}
                        # W index: 2*(2b+k') + i
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, r] += val
                        
            rank_idx += 1
            
    return {"U": U, "V": V, "W": W}
