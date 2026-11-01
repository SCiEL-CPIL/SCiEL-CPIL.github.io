"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
                r += 1
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T4 = T.reshape(n, m, m, p, p, n)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (j == jp and i == ip and k == kp) else 0
                                if abs(T4[i, j, jp, k, kp, ip] - expected) > 1e-6:
                                    return False
        return True
    
    best_rank = rank_trivial
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    # Try to find lower rank decompositions
    # Try ranks from 51 down to 35
    for target_rank in range(51, 34, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            np.random.seed(attempt * 7919 + target_rank * 131)
            U = np.random.randint(-3, 4, size=(n * m, target_rank))
            V = np.random.randint(-3, 4, size=(m * p, target_rank))
            W = np.random.randint(-3, 4, size=(p * n, target_rank))
            
            if check_decomposition(U, V, W):
                found = True
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
