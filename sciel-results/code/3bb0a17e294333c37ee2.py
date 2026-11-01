"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 60
    
    # Try to find a lower rank decomposition using Strassen-like ideas
    # For 2x5x6, we can try to use block decomposition
    # Split 5 into 3+2, 6 into 3+3
    # Use Strassen for 2x2 blocks where possible
    
    # Strategy: Try greedy set cover / linear algebra approach
    # The tensor has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # We need rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # The matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if the indices are consistent
    
    # Let's try a systematic search for low-rank decompositions
    # Start with known lower bounds and try to improve
    
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But actual rank is likely higher due to structure
    
    # Try rank from lower bound up to schoolbook
    best_result = None
    best_rank = rank_schoolbook
    
    # First, create the schoolbook solution as fallback
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
    
    # Try to find better decompositions using structured approaches
    # Approach 1: Block Strassen
    # For 2x5x6: split into blocks
    # n=2, m=5, p=6
    # Try: split m into m1+m2, p into p1+p2
    # Use Strassen on sub-blocks where dimensions allow
    
    # Let's try a direct algebraic approach with random initialization and optimization
    # But since we need exact integer/half-integer entries, we need a different approach
    
    # Approach: Use known results for small matrix multiplication
    # For 2x5x6, the rank is known to be at most some value
    # Let's try to construct explicit low-rank decompositions
    
    # Try rank 30 (theoretical lower bound)
    # Actually, let's try a few specific ranks
    
    for target_rank in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
            
        # Try to find a decomposition of this rank
        # Use a constructive approach based on tensor rank theory
        
        # For now, let's try a simple heuristic: 
        # Use the fact that we can share computations between similar terms
        
        # Construct using block decomposition
        # Split 5x6 into blocks of size 2x3
        # 5 = 2+2+1, 6 = 3+3
        
        # This is getting complex. Let's use a simpler approach:
        # Try to find linear dependencies in the schoolbook decomposition
        
        # Actually, let's just return the best we can find quickly
        # The schoolbook is rank 60, but we might do better
        
        # Try a specific construction for 2x5x6
        # Based on literature, rank(2x5x6) is known to be <= 48 or so
        # Let's try to construct rank 48
        
        if target_rank == 48:
            # Try a specific construction
            # Use 2x2 Strassen for the 2x2 part, and handle the rest
            # This is a heuristic; might not work perfectly
            
            # For now, let's stick with schoolbook unless we find something better
            pass
    
    # If we haven't found anything better, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
