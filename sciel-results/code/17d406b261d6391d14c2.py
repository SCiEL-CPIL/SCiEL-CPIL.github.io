"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank by checking for linear dependencies
    # Use SVD to find if any column is nearly dependent
    deadline = time.time() + 120
    
    # Try to reduce rank by checking subsets
    # For each column, check if it's in the span of others
    # This is expensive, so do a limited search
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    # Try removing columns one by one, starting from the end
    # Check if the tensor can still be represented
    # A column c is redundant if there exist alpha, beta, gamma such that
    # U[:,c] = alpha, V[:,c] = beta, W[:,c] = gamma and the outer products match
    # This is hard to check directly.
    
    # Alternative: use the fact that if rank r is achievable, we can try to find it
    # by solving a system of equations.
    
    # For now, return schoolbook as a safe fallback
    # Convert to lists
    U_list = [[float(x) for x in row] for row in current_U]
    V_list = [[float(x) for x in row] for row in current_V]
    W_list = [[float(x) for x in row] for row in current_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
