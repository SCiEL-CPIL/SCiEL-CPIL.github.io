import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create indices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We can use np.indices to create the tensor structure
    # But for rank-1 decomposition, we just need the basis vectors.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill using vectorized operations if possible, or just efficient loops
    # Since n, m, p are small, loops are fine, but let's make it "different"
    # by using a flat index mapping.
    
    for idx in range(rank):
        # Decompose idx into i, j, k
        # idx = i * (m * p) + j * p + k
        k = idx % p
        rem = idx // p
        j = rem % m
        i = rem // m
        
        # Set the corresponding entries
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, idx] = 1.0
        V[v_idx, idx] = 1.0
        W[w_idx, idx] = 1.0
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
