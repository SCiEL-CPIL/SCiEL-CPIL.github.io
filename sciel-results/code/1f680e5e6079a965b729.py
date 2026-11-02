import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[i,j,k] = 1 for the multiplication tensor
    # T is a 3D array where T[i,j,k] = 1 (the coefficient)
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U shape: (nm, rank), V shape: (mp, rank), W shape: (pn, rank)
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, rank):
        """Check using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try known constructions first: rank 49 for 5x5 matrix multiplication
    # The 5x5 matrix multiplication can be done with rank 49 using specific patterns
    
    # Strategy: Try ranks from 49 upward, using random initialization + local search
    
    best_result = None
    
    # First, try the schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better ranks
    # Known: 5x5 matrix multiplication has rank at most 49
    # Let's try random search for rank 49, 50, ..., up to some limit
    
    for target_rank in [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(target_rank)] for _ in range(pn)]
            
            if check_decomposition(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if best_result["U"] is not U_school:
            break
    
    # Try a more structured approach: use the known 5x5 construction
    # For 5x5, there's a construction based on grouping
    
    return best_result
