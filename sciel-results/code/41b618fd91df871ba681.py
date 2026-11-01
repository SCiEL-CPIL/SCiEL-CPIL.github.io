import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # Trivial decomposition
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_rank = trivial_rank
    best_result = {"U": U, "V": V, "W": W}

    # Strategy: Try to find lower-rank decompositions by exploiting structure
    # For matrix multiplication C = A * B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 (the "multiplication tensor")
    # We need: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k
    
    # Key insight: We can use the fact that for fixed j, the inner product
    # over j can be factored. Let's try rank = n*p + m*p + n*m - n*m*p/something
    
    # Actually, let's try a more systematic approach:
    # For each pair (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k]
    # This is a bilinear form in j. We can try to use fewer multiplications
    # by grouping terms.
    
    # Winograd-style: For 2x3x7, try rank 35, 30, 28, etc.
    # The minimum rank for matrix multiplication is at least max(n*p, m*p, n*m)
    # Lower bound: max(14, 21, 6) = 21
    
    # Let's try to construct explicit decompositions for various ranks
    
    # Approach 1: Use the fact that we can share computations across different (i,k) pairs
    # For fixed j, A[i,j] * B[j,k] contributes to C[i,k]
    # If we can express some A[i,j] or B[j,k] as linear combinations, we can reduce rank
    
    # Let's try a structured search:
    # For rank r, we need U (6×r), V (21×r), W (14×r)
    # with integer/half-integer entries such that the tensor equation holds
    
    # Try specific ranks from high to low
    for target_rank in range(trivial_rank - 1, max(n*p, m*p, n*m) - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Use a greedy/structured approach
        # Start with trivial decomposition and try to merge columns
        
        # Initialize with trivial rank-42 decomposition
        U_t = [[0] * trivial_rank for _ in range(n * m)]
        V_t = [[0] * trivial_rank for _ in range(m * p)]
        W_t = [[0] * trivial_rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i * m + j][col] = 1
                    V_t[j * p + k][col] = 1
                    W_t[k * n + i][col] = 1
                    col += 1
        
        # Try to merge pairs of columns where possible
        # Two columns (a, b, c) and (a', b', c') can be merged if there exists
        # alpha, beta such that alpha*a + beta*a' = 0, etc. for all relevant indices
        
        # This is complex, so let's try a different approach:
        # Use random projections and check validity
        
        # For small ranks, try random initialization with values in {-1, 0, 1, 0.5}
        values = [-1, -0.5, 0, 0.5, 1]
        
        found = False
        max_attempts = 1000 if target_rank > 30 else 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_r = [[random.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
            V_r = [[random.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
            W_r = [[random.choice(values) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(target_rank):
                            s += U_r[i * m + j][t] * V_r[j * p + k][t] * W_r[k * n + i][t]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                found = True
                break
        
        if found:
            break
    
    return best_result
