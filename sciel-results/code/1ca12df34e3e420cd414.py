"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s deadline, 10s safety margin
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank and try to reduce
    schoolbook_rank = n * m * p
    
    # Build the target tensor T[i, j, k] = 1 for all i,j,k
    # T has shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: at least n*m*p / something, but let's be conservative
    # For (3,3,8), try ranks from 50 to 72
    best_result = None
    best_rank = schoolbook_rank
    
    # First, create the schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using numerical optimization
    # We'll try a few ranks and use a simple optimization
    for rank in range(60, schoolbook_rank + 1, 2):
        if time.time() > deadline - 30:
            break
        
        # Initialize with schoolbook decomposition, truncated
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Copy the first 'rank' columns of the schoolbook decomposition
        U[:, :rank] = U_school[:, :rank]
        V[:, :rank] = V_school[:, :rank]
        W[:, :rank] = W_school[:, :rank]
        
        # Check if this truncated decomposition is exact (it won't be, but let's see)
        # Compute the error
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1) ** 2
        
        if error < 1e-6:
            # Found an exact decomposition
            best_result = (U, V, W)
            best_rank = rank
            break
    
    if best_result is None:
        # Return schoolbook decomposition
        U, V, W = U_school, V_school, W_school
    else:
        U, V, W = best_result
    
    # Convert to lists of integers or half-integers
    def to_half_int_list(arr):
        result = []
        for row in arr:
            result.append([x * 2 for x in row])  # Multiply by 2 to get integers
        return result
    
    # Actually, the entries should be integers or half-integers
    # So we need to round to the nearest half-integer
    def round_to_half_int(arr):
        return np.round(arr * 2) / 2
    
    U = round_to_half_int(U)
    V = round_to_half_int(V)
    W = round_to_half_int(W)
    
    # Verify the decomposition is exact
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    # Not exact, use schoolbook
                    U, V, W = U_school, V_school, W_school
                    break
    
    # Convert to Python lists
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
