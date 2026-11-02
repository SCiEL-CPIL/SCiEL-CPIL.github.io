"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
    def schoolbook():
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

    def verify_numpy(U, V, W):
        """Vectorized verification using numpy."""
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            if not np.allclose(T, 1.0, atol=1e-9):
                return False
            return True
        except Exception:
            return False

    best_result = schoolbook()
    best_rank = n * m * p

    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Upper bound from schoolbook: 3*4*7 = 84
    
    # Strategy: Use structured random search with Kronecker product ideas
    # Value pool for half-integer entries (must be integers or half-integers)
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2]
    
    # Try ranks from 28 up to 83, with more attempts for lower ranks
    for target_rank in range(28, best_rank):
        if time.time() > deadline:
            break
        
        # Limit attempts per rank based on remaining time
        remaining_time = deadline - time.time()
        if remaining_time < 1:
            break
        
        # Allocate attempts based on rank - lower ranks get more attempts
        if target_rank < 35:
            max_attempts = min(20, int(remaining_time * 0.5))
        elif target_rank < 50:
            max_attempts = min(10, int(remaining_time * 0.3))
        elif target_rank < 70:
            max_attempts = min(5, int(remaining_time * 0.2))
        else:
            max_attempts = min(3, int(remaining_time * 0.1))
        
        if max_attempts < 1:
            max_attempts = 1
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use deterministic seeding for reproducibility
            random.seed(hash((target_rank, attempt)) % (2**32))
            
            # Generate random matrices with values from the pool
            U = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify the decomposition
            if verify_numpy(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Move to next rank
    
    return best_result
# EVOLVE-BLOCK-END
