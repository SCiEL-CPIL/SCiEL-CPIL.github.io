"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    schoolbook_rank = n * m * p  # 60
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower-rank solutions
    # The equation: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For a given rank r, we have 56r unknowns and 60 equations
    # Try small ranks first
    
    # Try rank 30, 35, 40, etc.
    for rank in [30, 35, 40, 45, 50, 55]:
        if time.time() > deadline:
            break
        
        # Try to find a solution
        # Use a simple randomized approach
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with random small values
            np.random.seed(42 + rank + attempt)
            U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
            
            # Check if the decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to list of lists with half-integers
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
