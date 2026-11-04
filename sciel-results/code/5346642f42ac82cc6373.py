"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Construct Schoolbook Decomposition
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # We use numpy arrays for efficient construction, then convert to lists for output if required.
    # The prompt says "factor matrices of shapes...". Lists are safer for JSON serialization in some evaluators.
    
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
                
    # Verify the decomposition quickly to ensure correctness
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] should be 1 for all i,j,k
    # We check a few points or all points (120 points is fast)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_school[i*m+j, :] * V_school[j*p+k, :] * W_school[k*n+i, :])
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # This should never happen for schoolbook
        raise ValueError("Schoolbook decomposition failed verification")

    # No known efficient method to find rank < 120 for 4x5x6 within time limits.
    # Return the schoolbook solution.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
