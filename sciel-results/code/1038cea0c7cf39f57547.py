"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = 2, 2, 4
    rank = 4
    
    U = np.zeros((n*m, rank))  # shape (4, 4)
    V = np.zeros((m*p, rank))  # shape (8, 4)
    W = np.zeros((p*n, rank))  # shape (8, 4)
    
    # Term 0: A[0,0] * B[0,:] -> covers C[0,0..3] from j=0
    # U[i*m+j, c] = U[0*2+0, 0] = U[0, 0]
    U[0, 0] = 1
    # V[j*p+k, c] for j=0, k=0..3: V[0,0], V[1,0], V[2,0], V[3,0]
    for k in range(p):
        V[0 * p + k, 0] = 1
    # W[k*n+i, c] for k=0..3, i=0: W[0,0], W[2,0], W[4,0], W[6,0]
    for k in range(p):
        W[k * n + 0, 0] = 1
    
    # Term 1: A[0,1] * B[1,:] -> covers C[0,0..3] from j=1
    # U[0*2+1, 1] = U[1, 1]
    U[1, 1] = 1
    # V[1*p+k, 1] for k=0..3: V[4,1], V[5,1], V[6,1], V[7,1]
    for k in range(p):
        V[1 * p + k, 1] = 1
    # W[k*n+0, 1] for k=0..3: W[0,1], W[2,1], W[4,1], W[6,1]
    for k in range(p):
        W[k * n + 0, 1] = 1
    
    # Term 2: A[1,0] * B[0,:] -> covers C[1,0..3] from j=0
    # U[1*2+0, 2] = U[2, 2]
    U[2, 2] = 1
    # V[0*p+k, 2] for k=0..3: V[0,2], V[1,2], V[2,2], V[3,2]
    for k in range(p):
        V[0 * p + k, 2] = 1
    # W[k*n+1, 2] for k=0..3: W[1,2], W[3,2], W[5,2], W[7,2]
    for k in range(p):
        W[k * n + 1, 2] = 1
    
    # Term 3: A[1,1] * B[1,:] -> covers C[1,0..3] from j=1
    # U[1*2+1, 3] = U[3, 3]
    U[3, 3] = 1
    # V[1*p+k, 3] for k=0..3: V[4,3], V[5,3], V[6,3], V[7,3]
    for k in range(p):
        V[1 * p + k, 3] = 1
    # W[k*n+1, 3] for k=0..3: W[1,3], W[3,3], W[5,3], W[7,3]
    for k in range(p):
        W[k * n + 1, 3] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
