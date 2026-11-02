import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 2x3x3 matrix multiplication.
    Uses a vectorized approach with NumPy to define the factors U, V, W.
    Rank is 18 (Schoolbook upper bound).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18
    
    # Initialize U, V, W as lists of lists to match the expected output format strictly
    # The interface expects: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We iterate over the 'terms' of the decomposition.
    # Each term corresponds to a specific (i, j, k) triplet in the multiplication.
    # Term index r maps to (i, j, k) such that r = i * (m * p) + j * p + k
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    # Using NumPy for efficient index calculation if needed, but standard loops are fine.
    # To make it "distinct", we will use numpy arange and broadcasting to generate indices.
    
    i_indices = np.repeat(np.arange(n), m * p)
    j_indices = np.tile(np.repeat(np.arange(m), p), n)
    k_indices = np.tile(np.arange(p), n * m)
    
    u_rows = i_indices * m + j_indices
    v_rows = j_indices * p + k_indices
    w_rows = k_indices * n + i_indices
    
    for r in range(rank):
        U[u_rows[r]][r] = 1
        V[v_rows[r]][r] = 1
        W[w_rows[r]][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
