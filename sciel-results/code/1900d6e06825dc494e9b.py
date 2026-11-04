import numpy as np

def _schoolbook_decomp(n, m, p):
    """
    Returns U, V, W for M_{n,m,p} using schoolbook.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 2, 5, 5
    
    # Approach: Block Decomposition
    # Split p into p1 and p2
    p1, p2 = 2, 3
    
    # Decompose M_{n, m, p1}
    U1, V1, W1 = _schoolbook_decomp(n, m, p1)
    
    # Decompose M_{n, m, p2}
    # Note: The V and W indices need to be mapped correctly for the full p dimension.
    # V for the second block corresponds to columns p1..p-1 of B.
    # W for the second block corresponds to columns p1..p-1 of C.
    
    # We need to construct the full U, V, W for M_{n, m, p}
    # U is shared? No, U depends on i, j. It is the same for both blocks.
    # But in the tensor decomposition, each term is a rank-1 tensor.
    # The terms for block 1 use V indices j*p + k for k in 0..p1-1.
    # The terms for block 2 use V indices j*p + k for k in p1..p-1.
    
    # So we can just concatenate the columns of U, V, W.
    # But we must shift the indices in V and W for the second block.
    
    # Let's create the full matrices
    rank1 = U1.shape[1]
    rank2 = U1.shape[1] # Same dimensions n, m, so same rank? 
    # Wait, _schoolbook_decomp(n, m, p2) will have rank n*m*p2.
    # U1 shape: (n*m, n*m*p1)
    # U2 shape: (n*m, n*m*p2)
    
    U2, V2, W2 = _schoolbook_decomp(n, m, p2)
    
    # U is simply concatenated
    U_full = np.hstack([U1, U2])
    
    # V needs index shifting for the second block.
    # V1 indices are j*p1 + k.
    # In the full V of size m*p, the indices for block 1 are j*p + k.
    # Since p1 < p, the indices j*p1 + k are NOT the same as j*p + k.
    # We need to map the local indices to global indices.
    
    V_full = np.zeros((m * p, rank1 + rank2), dtype=float)
    W_full = np.zeros((p * n, rank1 + rank2), dtype=float)
    
    # Fill V_full for block 1
    for j in range(m):
        for k in range(p1):
            local_idx = j * p1 + k
            global_idx = j * p + k
            V_full[global_idx, :rank1] = V1[local_idx, :]
            
    # Fill W_full for block 1
    for k in range(p1):
        for i in range(n):
            local_idx = k * n + i
            global_idx = k * n + i
            W_full[global_idx, :rank1] = W1[local_idx, :]
            
    # Fill V_full for block 2
    # Local indices in V2 are j*p2 + k (k in 0..p2-1)
    # Global indices in V_full are j*p + (p1 + k)
    for j in range(m):
        for k in range(p2):
            local_idx = j * p2 + k
            global_idx = j * p + (p1 + k)
            V_full[global_idx, rank1:rank1+rank2] = V2[local_idx, :]
            
    # Fill W_full for block 2
    # Local indices in W2 are k*n + i (k in 0..p2-1)
    # Global indices in W_full are (p1 + k)*n + i
    for k in range(p2):
        for i in range(n):
            local_idx = k * n + i
            global_idx = (p1 + k) * n + i
            W_full[global_idx, rank1:rank1+rank2] = W2[local_idx, :]
            
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
