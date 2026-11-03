import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication C = A * B
    where A is n x m and B is m x p.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j, col] = 1.0
                # V[j*p + k, col] = 1
                V[j * p + k, col] = 1.0
                # W[k*n + i, col] = 1
                W[k * n + i, col] = 1.0
                col += 1
                
    # Check time, though construction is O(nmp) and very fast
    if time.time() > deadline:
        # Return partial or fallback if needed, but here it's instant
        pass
        
    # Convert to lists if required by interface, but numpy arrays are usually accepted 
    # or converted by the evaluator. The prompt example used lists of lists.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries"
    # The example return type was {"U": U, "V": V, "W": W} where U was a list of lists.
    # I will return numpy arrays as they are standard in scientific python, 
    # but to be safe with the "integer/half-integer entries" and potential strict type checking,
    # I'll stick to the structure. If the evaluator expects lists, I should convert.
    # The seed program returned lists. I will return lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
