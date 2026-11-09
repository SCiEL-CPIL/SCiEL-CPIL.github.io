"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook decomposition as fallback
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
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The tensor equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try rank from best_rank-1 down to some minimum
    for target_rank in range(best_rank - 1, max(20, best_rank - 40) - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with schoolbook and zero out last (best_rank - target_rank) columns
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Fill with schoolbook pattern for first target_rank terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
        
        # Now we need to adjust to satisfy all 60 equations
        # This is underdetermined if target_rank is close to 60
        # Let's try a simple iterative approach
        
        # Actually, let's try a different approach: use the fact that we can
        # express the missing terms as linear combinations
        
        # For now, just return the best we have
        if time.time() > deadline - 5:
            break
    
    return best
# EVOLVE-BLOCK-END
