import numpy as np

def construct():
    n, m, p = 2, 3, 7
    
    # We will build the decomposition by concatenating blocks.
    # Block 1: 2x3 by 3x3 (cols 0,1,2)
    # Block 2: 2x3 by 3x3 (cols 3,4,5)
    # Block 3: 2x3 by 3x1 (col 6)
    
    rank1 = 18 # Schoolbook for 2x3 by 3x3
    rank2 = 18 # Schoolbook for 2x3 by 3x3
    rank3 = 3  # Schoolbook for 2x3 by 3x1
    
    total_rank = rank1 + rank2 + rank3
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Helper to add a block
    def add_block(col_start, width, rank_offset):
        # col_start: index of first column in B (0..p-1)
        # width: number of columns
        # rank_offset: offset in U,V,W columns
        
        # U indices: i*m + j (0..n*m-1)
        # V indices: j*p + k (0..m*p-1) -> but k is local to block?
        # The problem defines V[j*p + k, c] where k is 0..p-1.
        # So for block starting at col_start, the V index is j*p + (col_start + k_local).
        # W indices: k*n + i (0..p*n-1) -> k is global col index.
        
        for i in range(n):
            for j in range(m):
                for k_local in range(width):
                    k_global = col_start + k_local
                    c = rank_offset + i*m + j + k_local # This is wrong, c is the column in the decomposition
                    # The decomposition column index c is just the index in the rank dimension.
                    # We need to map (i, j, k_local) to a unique c in [0, rank1)
                    # Let's just use a counter.
                    pass
        
    # Let's just explicitly build the 39 rank decomposition.
    
    # Block 1: cols 0,1,2
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # 0,1,2
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    # c is now 18
    
    # Block 2: cols 3,4,5
    for i in range(n):
        for j in range(m):
            for k in range(3): # 3,4,5
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    # c is now 36
    
    # Block 3: col 6
    for i in range(n):
        for j in range(m):
            for k in range(1): # 6
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    # c is now 39
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
