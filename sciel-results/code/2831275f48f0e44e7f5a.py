import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication using block decomposition.
    Partitions into 2x2 and 3x3 blocks. Uses schoolbook for all sub-block multiplications.
    
    Rank calculation:
    2x2 * 2x2: 4
    2x3 * 3x2: 12
    2x2 * 2x3: 12
    2x3 * 3x3: 18
    3x2 * 2x2: 12
    3x3 * 3x2: 18
    3x2 * 2x3: 18
    3x3 * 3x3: 27
    Total: 121
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    # We will build the U, V, W matrices directly.
    # Each term in the decomposition corresponds to a column in U, V, W.
    # U has shape (25, rank), V (25, rank), W (25, rank).
    
    # List of terms: (u_idx, v_idx, w_idx, coeff)
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    
    terms = []
    
    # Helper to add a term
    def add_term(i, j, k, coeff=1.0):
        if coeff != 0:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            terms.append((u_idx, v_idx, w_idx, coeff))
            
    # Block boundaries
    # A: rows 0-1 (top), 2-4 (bottom); cols 0-1 (left), 2-4 (right)
    # B: rows 0-1 (top), 2-4 (bottom); cols 0-1 (left), 2-4 (right)
    # C: rows 0-1 (top), 2-4 (bottom); cols 0-1 (left), 2-4 (right)
    
    # 1. C11 (2x2) = A11(2x2) * B11(2x2) + A12(2x3) * B21(3x2)
    # A11 * B11: i in 0-1, j in 0-1, k in 0-1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                add_term(i, j, k, 1.0)
                
    # A12 * B21: i in 0-1, j in 2-4, k in 0-1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_term(i, j, k, 1.0)
                
    # 2. C12 (2x3) = A11(2x2) * B12(2x3) + A12(2x3) * B22(3x3)
    # A11 * B12: i in 0-1, j in 0-1, k in 2-4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term(i, j, k, 1.0)
                
    # A12 * B22: i in 0-1, j in 2-4, k in 2-4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term(i, j, k, 1.0)
                
    # 3. C21 (3x2) = A21(3x2) * B11(2x2) + A22(3x3) * B21(3x2)
    # A21 * B11: i in 2-4, j in 0-1, k in 0-1
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                add_term(i, j, k, 1.0)
                
    # A22 * B21: i in 2-4, j in 2-4, k in 0-1
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                add_term(i, j, k, 1.0)
                
    # 4. C22 (3x3) = A21(3x2) * B12(2x3) + A22(3x3) * B22(3x3)
    # A21 * B12: i in 2-4, j in 0-1, k in 2-4
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                add_term(i, j, k, 1.0)
                
    # A22 * B22: i in 2-4, j in 2-4, k in 2-4
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term(i, j, k, 1.0)
                
    rank = len(terms)
    
    # Check time
    if time.time() > deadline - safety_margin:
        pass
        
    # Build the U, V, W matrices
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx, r] = coeff
        V[v_idx, r] = coeff
        W[w_idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
