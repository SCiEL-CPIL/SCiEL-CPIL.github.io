"""Matrix multiplication 2,3,9 - rank-minimal tensor decomposition."""
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # Block structure: p=9 = 3 blocks of size 3
    block_size = 3
    num_blocks = p // block_size  # = 3
    
    # For each [2,3,3] block, rank is 2 (using u=(1,0) and u=(0,1) with v=(1,1,1))
    # Total rank = 3 * 2 = 6
    total_rank = num_blocks * 2  # = 6
    
    # U: (n*m, rank) = (6, 6)
    # V: (m*p, rank) = (27, 6)
    # W: (p*n, rank) = (18, 6)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        # Block b covers k indices: 3*b, 3*b+1, 3*b+2
        k_start = b * block_size
        
        # Term 2*b: u=(1,0), v=(1,1,1), w has 1s at k_start..k_start+2
        col1 = 2 * b
        # U: u=(1,0) means U[0*3+j, col1]=1 for j=0,1,2 and U[1*3+j, col1]=0
        for j in range(m):
            U[0 * m + j][col1] = 1  # i=0
            U[1 * m + j][col1] = 0  # i=1
        # V: v=(1,1,1) means V[j*p+k, col1]=1 for all j, k in this block
        for j in range(m):
            for k in range(k_start, k_start + block_size):
                V[j * p + k][col1] = 1
        # W: w has 1s at positions k_start..k_start+2 for both i=0 and i=1
        for k in range(k_start, k_start + block_size):
            W[k * n + 0][col1] = 1  # i=0
            W[k * n + 1][col1] = 1  # i=1
        
        # Term 2*b+1: u=(0,1), v=(1,1,1), w has 1s at k_start..k_start+2
        col2 = 2 * b + 1
        for j in range(m):
            U[0 * m + j][col2] = 0  # i=0
            U[1 * m + j][col2] = 1  # i=1
        for j in range(m):
            for k in range(k_start, k_start + block_size):
                V[j * p + k][col2] = 1
        for k in range(k_start, k_start + block_size):
            W[k * n + 0][col2] = 1  # i=0
            W[k * n + 1][col2] = 1  # i=1
    
    return {"U": U, "V": V, "W": W}
