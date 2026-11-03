"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Trivial rank-48 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
    
    def is_valid(U, V, W):
        """Check if U, V, W form a valid decomposition"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Use a structured approach: try ranks from high to low
    # For each rank, try random initializations with small integer/half-integer values
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    
    np.random.seed(42)
    
    # Try ranks from 40 down to 20
    for target_rank in range(40, 19, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random attempts for this rank
        attempts = 100 if target_rank > 30 else 50
        
        for attempt in range(attempts):
            if time.time() > deadline - 15:
                break
            
            # Generate random half-integer entries with small magnitudes
            U = np.random.randint(-3, 4, (n*m, target_rank)) * 0.5
            V = np.random.randint(-3, 4, (m*p, target_rank)) * 0.5
            W = np.random.randint(-3, 4, (p*n, target_rank)) * 0.5
            
            if is_valid(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists for return
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
