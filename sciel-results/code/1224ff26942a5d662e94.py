import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 3x5 * 5x7.
    
    The matrix multiplication tensor T is defined such that:
    (A * B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
    
    In tensor notation, T = sum_{i,j,k} e_{i,j} (A) \otimes e_{j,k} (B) \otimes e_{k,i} (C)
    where A is n x m, B is m x p, C is p x n.
    
    The rank is n*m*p. We construct the factor matrices U, V, W.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    Indexing convention:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    
    We iterate over all (i, j, k) tuples. Each tuple corresponds to a rank-1 term.
    """
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using NumPy for efficient initialization, though loops are fine for small sizes
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row i*m + j
                u_idx = i * m + j
                # Index into V: row j*p + k
                v_idx = j * p + k
                # Index into W: row k*n + i
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Convert to lists as required by the interface (standard Python lists)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
