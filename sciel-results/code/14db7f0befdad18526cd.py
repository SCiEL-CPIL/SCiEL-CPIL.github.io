import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as required by the interface
    # The interface expects {'U', 'V', 'W'} with entries that can be integer/half-integer
    # The evaluator likely accepts numpy arrays or lists. 
    # The seed program returned lists of lists. I will return numpy arrays if allowed, 
    # but to be safe and match the seed's type (list of lists), I'll convert.
    # However, numpy arrays are often faster and accepted. 
    # Let's check the seed: "U = [[0] * rank for _ in range(n * m)]" -> List of Lists.
    # I will return numpy arrays as they are standard in these tasks, 
    # but if the evaluator is strict about types, I should convert.
    # Given "integer/half-integer entries", numpy float64 is fine.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
