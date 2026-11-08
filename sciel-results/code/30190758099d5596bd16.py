import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This is the standard bilinear form for matrix multiplication.
    # Rank = n * m * p
    n, m, p = DIMENSIONS
    
    # For 5x5x5, rank is 125.
    # We construct the factor matrices directly.
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists (or numpy arrays if preferred, but lists are fine for small sizes)
    # Using lists to avoid import overhead if not needed, but NumPy is allowed and might be faster for construction?
    # Given the small size (125 columns, 25 rows), pure Python lists are sufficient and fast enough.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m + j
                U[i * m + j][col] = 1
                # V index: row j*p + k
                V[j * p + k][col] = 1
                # W index: row k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
