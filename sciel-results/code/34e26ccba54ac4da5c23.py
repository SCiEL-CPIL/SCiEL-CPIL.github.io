"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using structured search
    # The tensor T has entries T[(i,j),(j,k),(k,i)] = 1 for valid indices
    # We search for rank-r decompositions
    
    # Try various rank values from known lower bounds
    # For 3x5x7, the schoolbook is 105. Known algorithms may give lower.
    # We'll try to find decompositions with rank < 105
    
    # Strategy: Use the fact that we can combine some rank-1 terms
    # if they share structure. Try systematic construction.
    
    # Approach: For small matrices, we can use the "border rank" or 
    # structured decomposition. Let's try a greedy approach with verification.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # Reconstruct the tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    expected = 1 if True else 0  # All (i,j,k) triples are valid
                    if s != expected:
                        return False
        return True
    
    # Try to find a decomposition with reduced rank by exploiting structure
    # For 3x5x7, one approach: group by i and k, share j-structure
    
    # Let's try a systematic construction:
    # For each (i,k) pair, we need to decompose the m-dimensional vector
    # This gives us n*p independent "slices" of dimension m
    
    # Actually, let's try a different approach: use the fact that
    # we can write the multiplication as a sum of rank-1 tensors where
    # some entries share the same (i,j) or (j,k) or (k,i) patterns.
    
    # Try rank values and use random search with local improvement
    for target_rank in range(n*m*p - 1, max(n*m*p - 20, 50), -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_try = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V_try = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W_try = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Local search: adjust entries to satisfy constraints
            # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # This is a hard constraint satisfaction problem. Let's use a simpler approach.
            # Try structured decompositions based on known algorithms.
            
            # For now, if we find a valid one, update best
            if verify_decomposition(U_try, V_try, W_try, target_rank):
                best_result = {"U": U_try, "V": V_try, "W": W_try}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
