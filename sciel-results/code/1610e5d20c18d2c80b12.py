"""Program for matrix multiplication tensor decomposition (4x5 * 5x6)."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    Returns dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (20, rank)
      V has shape (m*p, rank) = (30, rank)
      W has shape (p*n, rank) = (24, rank)
    
    The decomposition satisfies:
      (A @ B)[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for all i, k, summed over j.
    """
    n, m, p = DIMENSIONS
    
    # We want to find a rank-minimal decomposition.
    # The schoolbook rank is n*m*p = 120.
    # We'll try to find a smaller rank using a structured approach.
    
    # One approach: use the fact that matrix multiplication can be decomposed
    # using smaller building blocks. For example, we can decompose the 4x5*5x6
    # multiplication using 2x2*2x2 blocks.
    
    # Let's try a different approach: use a random search with small integer entries
    # to find a decomposition with rank < 120.
    
    # Actually, let's try a more systematic approach. We'll use the fact that
    # for rectangular matrix multiplication, there are known decompositions
    # that are better than schoolbook.
    
    # For 4x5 * 5x6, we can try to use a decomposition based on the
    # "Strassen-like" approach, but adapted for rectangular matrices.
    
    # Let's try to find a decomposition with rank = 60 (half of schoolbook).
    # We'll use a greedy algorithm to find such a decomposition.
    
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We need to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # This is a system of n*m*p = 120 equations in r * (n*m + m*p + p*n) = r * 74 unknowns.
    
    # Let's try to find a solution with small integer entries.
    
    # We'll use a random search approach.
    
    import random
    import time
    
    start_time = time.time()
    
    # Try different ranks from 30 (lower bound) up to 120 (schoolbook)
    for rank in range(30, 121):
        if time.time() - start_time > 100:
            break
        
        # Try to find a decomposition with this rank
        for attempt in range(1000):
            if time.time() - start_time > 100:
                break
            
            # Initialize U, V, W with small random integer entries
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            # Fill with random entries in {-1, 0, 1}
            for _ in range(n * m * rank):
                idx = random.randint(0, n * m * rank - 1)
                row = idx // rank
                col = idx % rank
                U[row, col] = random.choice([-1, 0, 1])
            
            for _ in range(m * p * rank):
                idx = random.randint(0, m * p * rank - 1)
                row = idx // rank
                col = idx % rank
                V[row, col] = random.choice([-1, 0, 1])
            
            for _ in range(p * n * rank):
                idx = random.randint(0, p * n * rank - 1)
                row = idx // rank
                col = idx % rank
                W[row, col] = random.choice([-1, 0, 1])
            
            # Check if this decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] 
                                  for c in range(rank))
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                # Convert to lists and return
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # If we couldn't find a smaller rank, return the schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
