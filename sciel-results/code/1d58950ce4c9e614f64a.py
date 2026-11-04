import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140 # Leave 10s for safety
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # Note: W index is k*n + i according to trace convention W[k*n+i, r]
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Target tensor T[i, j, k] = 1 if we are computing C[i, k] = sum A[i, j] B[j, k]
    # The decomposition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k?
    # No, the tensor for matrix multiplication is T[i, j, k] = 1.
    # Wait, the contraction is sum_j A[i,j] B[j,k] = C[i,k].
    # The tensor T is such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c
    # where U_c is vector of size n*m, V_c size m*p, W_c size p*n.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We want to find rank r such that this holds.
    
    # Strategy: Continuous relaxation + Rounding
    # We will try ranks from 74 down to 70.
    # For each rank, we do a few iterations of "approximate" search.
    
    for r in range(74, 69, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Simple local search / gradient step approximation
            # We want to minimize sum_{i,j,k} (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
            # This is expensive to compute fully, but n*m*p = 75. r is ~70.
            # 75 * 70 = 5250 ops per error calc. Very fast.
            
            # We will do a few steps of "coordinate descent" or just check if rounding works immediately
            # Given the difficulty of continuous optimization in 140s, we might just check
            # if a "structured" random search with half-integers works.
            
            # Let's try a discrete search with half-integers {-2, -1.5, ..., 2}
            # But that's 9 values. 9^70 is huge.
            
            # Better: Use the fact that schoolbook is 75.
            # Can we find a rank 74 solution by "merging" two schoolbook terms?
            # If U_a, V_a, W_a and U_b, V_b, W_b are schoolbook terms,
            # can we replace them with a single term? No, rank 1 terms are independent.
            
            # Let's try a specific structured search:
            # Assume the solution is close to schoolbook.
            # Start with schoolbook rank 75.
            # Try to eliminate one column by setting it to zero and adjusting others?
            # This is hard.
            
            # Alternative: Use the known Strassen decomposition for 2x2x2.
            # If we can embed a 2x2x2 Strassen block, we save 1 rank.
            # Total rank 74.
            
            # Let's try to construct a rank 74 solution manually using blocks.
            # Split the 5x5 multiplication into a 2x2 block and a 3x3 block?
            # As analyzed before, this sums to 75.
            
            # What if we use a 2x2 Strassen block for the 2x2 part of B, 
            # and standard for the rest?
            # This requires A to be split accordingly.
            
            # Let's try a random search with entries from {-1, 0, 1, 2} for rank 74.
            # This is a different distribution than the current {-2..2}.
            
            U_rand = np.random.randint(-2, 3, (n * m, r))
            V_rand = np.random.randint(-2, 3, (m * p, r))
            W_rand = np.random.randint(-2, 3, (p * n, r))
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                best_U, best_V, best_W, best_rank = U_rand, V_rand, W_rand, r
                break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists of lists for the output interface
    U_out = [[int(x) for x in row] for row in best_U]
    V_out = [[int(x) for x in row] for row in best_V]
    W_out = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
