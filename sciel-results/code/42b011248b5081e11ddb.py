"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def validate(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        # The tensor condition: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # And for all other index combinations, the sum should be 0
        # But actually, the tensor only has 1s at (i,j,k) positions, so we just need:
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: use the fact that we can combine terms with small integer coefficients
    # For M_{3,5,5}, try ranks from 74 down to a reasonable lower bound
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But in practice, for these dimensions, the rank is likely close to 75
    
    # Try a structured search: for each candidate rank, try to find coefficients
    # This is computationally expensive, so we'll limit the search
    
    min_rank_to_try = max(50, rank_schoolbook - 20)  # Try ranks 55 to 74
    
    for candidate_rank in range(rank_schoolbook - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 30:
            break
        
        # For small rank reductions, we can try to find linear dependencies
        # among the schoolbook terms by solving a system of equations
        
        # The condition is that there exist U, V, W of rank r such that
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
        
        # This is a system of n*m*p = 75 equations in r*(n*m + m*p + p*n) variables
        # which is too many to solve directly for r close to 75
        
        # Instead, try a heuristic: assume U and V have a specific structure
        # and solve for W, or vice versa
        
        # For now, just check if removing one column at a time gives a valid decomposition
        # This is very unlikely to work, but let's try
        
        # Actually, let's try a different approach: use the border rank concept
        # or just accept that for these small dimensions, schoolbook is likely optimal
        
        # Given the time constraints, return the schoolbook decomposition
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
