"""Matrix multiplication 2x4 by 4x8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 64
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    def validate(U_try, V_try, W_try, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Start from rank 56 and go down
    for target_rank in range(56, 47, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 104729)
            
            # Use entries from {-1, 0, 1} for sparsity and exactness
            U_try = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V_try = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W_try = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            if validate(U_try, V_try, W_try, target_rank):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break
    
    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
