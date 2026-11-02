import numpy as np

def construct():
    n, m, p = 2, 2, 15
    rank = 53
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen factors for 2x2x2
    # r: 0..6
    # U[i,j,r], V[j,k,r], W[k,i,r]
    # i,j,k in {0,1}
    
    # M1
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # M2
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1; W[1*2+1, 1] = -1
    
    # M3
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+1, 2] = 1
    
    # M4
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1
    
    # M5
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1
    
    # M6
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # M7
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    # Copy Strassen blocks for 7 groups of 2 columns
    for b in range(7):
        offset = b * 7
        for r in range(7):
            # U is same for all blocks (A is shared)
            # But we need to map local V and W to global indices
            # U index is local i,j -> global i,j. Since A is same, U col offset is just offset.
            # Wait, U shape is (4, 53). The columns correspond to the rank factors.
            # So for block b, we use columns offset..offset+6 of U, V, W.
            
            # U values are identical for all blocks
            # U[local_i_j, offset+r] = U_strassen[local_i_j, r]
            # But we already defined U for the first block in cols 0-6.
            # We just need to copy them to other offsets.
            U[:, offset+r] = U[:, r]
            
            # V: local j, k' -> global j, k = 2b + k'
            # V index = j * 15 + k
            # Local V index = j * 2 + k'
            for j in range(2):
                for k_prime in range(2):
                    v_local = j * 2 + k_prime
                    v_global = j * 15 + (2*b + k_prime)
                    V[v_global, offset+r] = V[v_local, r]
                    
            # W: local k', i -> global k = 2b + k', i
            # W index = k * 2 + i
            # Local W index = k' * 2 + i
            for k_prime in range(2):
                for i in range(2):
                    w_local = k_prime * 2 + i
                    w_global = (2*b + k_prime) * 2 + i
                    W[w_global, offset+r] = W[w_local, r]

    # Schoolbook for last block (1 column)
    # Offset = 49
    offset = 49
    # r=0: A00 B00 -> C00
    U[0*2+0, offset+0] = 1
    V[0*15+14, offset+0] = 1 # j=0, k=14
    W[14*2+0, offset+0] = 1  # k=14, i=0
    
    # r=1: A01 B10 -> C00
    U[0*2+1, offset+1] = 1
    V[1*15+14, offset+1] = 1 # j=1, k=14
    W[14*2+0, offset+1] = 1  # k=14, i=0
    
    # r=2: A10 B00 -> C10
    U[1*2+0, offset+2] = 1
    V[0*15+14, offset+2] = 1 # j=0, k=14
    W[14*2+1, offset+2] = 1  # k=14, i=1
    
    # r=3: A11 B10 -> C10
    U[1*2+1, offset+3] = 1
    V[1*15+14, offset+3] = 1 # j=1, k=14
    W[14*2+1, offset+3] = 1  # k=14, i=1

    return {'U': U, 'V': V, 'W': W}
