import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as in the seed, or numpy?
    # The seed used lists. The evaluator might expect lists or numpy.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # Usually numpy arrays are preferred for speed, but lists are fine for small sizes.
    # I will use numpy for robustness.
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Check deadline
    if time.time() > deadline:
        pass # This shouldn't happen for this small size
        
    return {"U": U, "V": V, "W": W}
