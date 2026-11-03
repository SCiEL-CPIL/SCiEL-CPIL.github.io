import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as NumPy arrays of zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Map rank index to (i, j, k)
    # We iterate through all i, j, k
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    # Convert to lists of lists to match the expected interface if strict types are required,
    # though numpy arrays are often accepted. The prompt example used lists.
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is usually fine for evaluators that convert or check values.
    # However, to be safe and match the "lists" style of the seed if it matters, 
    # we can return .tolist(). But numpy is faster to generate.
    # Let's return numpy arrays as they are standard in such tasks unless specified otherwise.
    # The seed returned lists. I will return lists to be perfectly compatible with the seed's interface style.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
