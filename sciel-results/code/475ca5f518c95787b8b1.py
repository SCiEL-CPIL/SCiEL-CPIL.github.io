import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a lower rank decomposition
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0
    
    # Build the list of non-zero tensor entries
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))
    
    # Try ranks from lower bound up to n*m*p
    # Lower bound: max(n*m, m*p, n*p)
    lower_bound = max(n * m, m * p, n * p)
    upper_bound = n * m * p
    
    # For each rank, try to find a solution
    # We'll use a simple approach: for rank r, we have r terms
    # T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    
    # This is a system of polynomial equations. We'll try to solve it
    # by fixing some variables and solving linear systems.
    
    # Strategy: For each rank r, try random initializations and use ALS
    # But for exact decomposition, we need to be more careful.
    
    # Let's try a different approach: use the fact that we can set
    # some entries of U, V, W to 1 or 0 to simplify.
    
    # For now, let's just return the schoolbook decomposition
    # as a fallback, but structure the code to attempt lower ranks.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
