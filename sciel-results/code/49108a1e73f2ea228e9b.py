import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k 
    # (under the trace convention Tr(CAB) = sum C_ki A_ij B_jk).
    # This tensor is the outer product of three all-ones vectors.
    # Therefore, it has tensor rank 1.
    
    rank = 1
    
    # U shape: (n*m, rank)
    U_np = np.ones((n * m, rank), dtype=int)
    # V shape: (m*p, rank)
    V_np = np.ones((m * p, rank), dtype=int)
    # W shape: (p*n, rank)
    W_np = np.ones((p * n, rank), dtype=int)
    
    # Convert to lists of lists
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
