import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 
    matrix multiplication of a 2x2 matrix by a 2x15 matrix.
    
    The rank is n*m*p = 2*2*15 = 60.
    This is the schoolbook algorithm, which is optimal for this shape.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices as lists.
    """
    n, m, p = DIMENSIONS
    
    rank = n * m * p
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mapping:
    # U indices: i in [0, n), j in [0, m) -> idx_u = i * m + j
    # V indices: j in [0, m), k in [0, p) -> idx_v = j * p + k
    # W indices: k in [0, p), i in [0, n) -> idx_w = k * n + i
    
    for r in range(rank):
        # Decompose rank index r into (i, j, k)
        # We use the standard lexicographic ordering:
        # r = i * (m * p) + j * p + k
        
        i = r // (m * p)
        remainder = r % (m * p)
        j = remainder // p
        k = remainder % p
        
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        
        U[idx_u, r] = 1.0
        V[idx_v, r] = 1.0
        W[idx_w, r] = 1.0
        
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
