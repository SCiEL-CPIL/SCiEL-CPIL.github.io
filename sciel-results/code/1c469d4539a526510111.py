import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Precompute target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                target_indices.append((a, b, c))
    
    def check_exact(U, V, W, rank):
        """Fast exact check."""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        # Compute result
        result = np.einsum('ar,br,cr->abc', U, V, W)
        # Check target entries
        for (a, b, c) in target_indices:
            if result[a, b, c] != 1:
                return False
        # Check non-target entries
        # Instead of iterating all, compute the full result and check
        # Non-target: result should be 0 where T is 0
        nonzero_non_target = np.count_nonzero(result * (1 - T))
        if nonzero_non_target > 0:
            return False
        return True
    
    random.seed(42)
    
    # Focus on plausible ranks
    for rank in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 10:
            break
        
        found = False
        # Adaptive number of trials based on time
        for trial in range(500):
            if time.time() > deadline - 10:
                break
            
            # Try different initialization strategies
            strategy = trial % 3
            
            if strategy == 0:
                # Random small integers
                U = np.random.randint(-2, 3, (n * m, rank)).astype(float)
                V = np.random.randint(-2, 3, (m * p, rank)).astype(float)
                W = np.random.randint(-2, 3, (p * n, rank)).astype(float)
            elif strategy == 1:
                # Start from scaled schoolbook-like
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = (i * m + j) * p + (j * p + k) - j * p + k  # messy
                            # Just use random assignment
                            r = random.randint(0, rank - 1)
                            U[i * m + j, r] += 1
                            V[j * p + k, r] += 1
                            W[k * n + i, r] += 1
            else:
                # Sparse random
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for r in range(rank):
                    for _ in range(random.randint(1, 3)):
                        a = random.randint(0, n * m - 1)
                        U[a, r] = random.choice([-1, 1, 2, -2])
                    for _ in range(random.randint(1, 3)):
                        b = random.randint(0, m * p - 1)
                        V[b, r] = random.choice([-1, 1, 2, -2])
                    for _ in range(random.randint(1, 3)):
                        c = random.randint(0, p * n - 1)
                        W[c, r] = random.choice([-1, 1, 2, -2])
            
            # Local search
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                if check_exact(U, V, W, rank):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = rank
                    found = True
                    break
                
                # Random perturbation
                choice = random.random()
                if choice < 0.33:
                    a = random.randint(0, n * m - 1)
                    r = random.randint(0, rank - 1)
                    U[a, r] += random.choice([-1, 1])
                elif choice < 0.66:
                    b = random.randint(0, m * p - 1)
                    r = random.randint(0, rank - 1)
                    V[b, r] += random.choice([-1, 1])
                else:
                    c = random.randint(0, p * n - 1)
                    r = random.randint(0, rank - 1)
                    W[c, r] += random.choice([-1, 1])
            
            if found:
                break
        
        if found:
            break
    
    return best_result
