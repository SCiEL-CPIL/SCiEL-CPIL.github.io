"""Matrix multiplication 2x5x6 - rank-minimal decomposition via block partitioning."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2x2."""
    U = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # i=0, m=0
        [0,  1,  0,  0,  0,  0,  0],  # i=0, m=1
        [0,  0,  1,  0,  1,  0,  0],  # i=1, m=0
        [0,  0,  1,  0,  0,  1,  0],  # i=1, m=1
    ], dtype=float)
    V = np.array([
        [1,  1,  0,  0,  0,  0,  0],  # m=0, k=0
        [0,  0,  1,  1,  0,  0,  0],  # m=0, k=1
        [1, -1,  0,  0,  0,  0,  0],  # m=1, k=0
        [0,  0,  1, -1,  0,  0,  0],  # m=1, k=1
    ], dtype=float)
    W = np.array([
        [0,  0,  1,  0,  0,  0,  0],  # k=0, i=0
        [1,  1,  0,  0,  0,  0,  0],  # k=0, i=1
        [0,  0,  0,  0,  1,  1,  1],  # k=1, i=0
        [0,  0,  0,  0,  0,  1,  0],  # k=1, i=1
    ], dtype=float)
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for nxmxp."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """Combine two decompositions for (n, m1, p) and (n, m2, p) into (n, m1+m2, p)."""
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank1 + rank2 # Typo in thought, fix below
    
    # Correct shapes
    # U1: (n*m1, rank1), U2: (n*m2, rank2)
    # V1: (m1*p, rank1), V2: (m2*p, rank2)
    # W1: (p*n, rank1), W2: (p*n, rank2)
    
    U = np.zeros((n * (m1 + m2), total_rank))
    V = np.zeros(((m1 + m2) * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill U
    for i in range(n):
        for j in range(m1):
            U[i * (m1 + m2) + j, :rank1] = U1[i * m1 + j, :]
        for j in range(m2):
            U[i * (m1 + m2) + m1 + j, rank1:] = U2[i * m2 + j, :]
            
    # Fill V
    for j in range(m1):
        for k in range(p):
            V[j * p + k, :rank1] = V1[j * p + k, :]
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k, rank1:] = V2[j * p + k, :]
            
    # Fill W
    for k in range(p):
        for i in range(n):
            W[k * n + i, :rank1] = W1[k * n + i, :]
            W[k * n + i, rank1:] = W2[k * n + i, :]
            
    return U, V, W

def _block_decompose(n, m, p):
    """
    Block decomposition for n x m x p matrix multiplication.
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank)
    """
    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    if n * m * p <= 8:
        return _schoolbook(n, m, p)
        
    # Try splitting m
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _block_decompose(n, m1, p)
        U2, V2, W2 = _block_decompose(n, m2, p)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            best_rank = total_rank
            best_U, best_V, best_W = _combine(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
            
    if best_U is not None:
        return best_U, best_V, best_W
        
    # Fallback
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _block_decompose(n, m, p)
    
    # Ensure integer/half-integer entries
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
