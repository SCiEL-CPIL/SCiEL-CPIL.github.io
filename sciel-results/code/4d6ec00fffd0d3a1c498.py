"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Initialize with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition is exact using vectorized numpy."""
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy 1: Try to find a lower-rank decomposition via structured search
    # For 3x3 * 3x6, we can think of it as 3x3 * (3x3 + 3x3) potentially
    # Or use the fact that 3x3 matmul has known rank ~23
    
    # Strategy 2: Random search with better initialization
    # Use the structure: each term in the sum contributes to specific (i,j,k) entries
    
    # Try ranks from a reasonable starting point downward
    # For 3x3*3x6, schoolbook is 54. Let's try to get below that.
    # Start from rank 50 and go down to maybe 30
    
    for rank in range(50, 25, -1):
        if time.time() > deadline - 15:
            break
            
        # Try several random initializations with different strategies
        max_attempts = 3 if rank > 40 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                    break
                
            rng = np.random.RandomState(attempt * 7919 + rank * 31)
            
            # Strategy: sparse random matrices with values from {-1, 0, 1}
            density = 0.3 if rank > 40 else 0.4
            U = rng.choice([-1, 0, 1], size=(n*m, rank))
            V = rng.choice([-1, 0, 1], size=(m*p, rank))
            W = rng.choice([-1, 0, 1], size=(p*n, rank))
            
            # Make them sparser
            mask_u = rng.random((n*m, rank)) < density
            mask_v = rng.random((m*p, rank)) < density
            mask_w = rng.random((p*n, rank)) < density
            U = U * mask_u
            V = V * mask_v
            W = W * mask_w
            
            # Verify
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank, n, m, p):
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        # If we found a valid decomposition at this rank, try lower
        if best_rank == rank:
            continue
    
    # Strategy 3: Local search - start from schoolbook and try to eliminate terms
    # This is more complex but might find structured lower-rank decompositions
    
    # Strategy 4: Use known results for 3x3 matmul
    # The 3x3 * 3x3 matmul tensor has rank at most 23 (Winograd) 
    # For 3x3 * 3x6, we can potentially use a similar approach
    
    # Let's try a more systematic approach: 
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,6)
    # We want to write this as sum of rank-1 tensors
    
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i]
    # where U is 3x3, V is 3x6, W is 6x3
    
    # Let's try to find a decomposition with rank around 30-40
    # using a more careful search
    
    if time.time() < deadline - 30:
        # Try a grid search over small integer/half-integer values
        # for a moderate rank
        for rank in [40, 38, 36, 34, 32, 30]:
            if time.time() > deadline - 10:
                break
                
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                    
                rng = np.random.RandomState(attempt * 104729 + rank * 13)
                
                # Use values from {-1, -0.5, 0, 0.5, 1}
                vals = [-1, -0.5, 0, 0, 0, 0.5, 1]
                U = rng.choice(vals, size=(n*m, rank))
                V = rng.choice(vals, size=(m*p, rank))
                W = rng.choice(vals, size=(p*n, rank))
                
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    break
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
