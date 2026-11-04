"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build the multiplication tensor T[i,j,k] = 1 for all valid indices
    # We want T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try various ranks and find the smallest that works
    best_result = None
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Use the schoolbook decomposition which guarantees integer (0 or 1) entries.
    # Rank is n*m*p.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    U = np.zeros((n*m, rank_school), dtype=int)
    V = np.zeros((m*p, rank_school), dtype=int)
    W = np.zeros((p*n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Convert to list format
    U_list = [[int(best_U[i,j]) for j in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i,j]) for j in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i,j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
