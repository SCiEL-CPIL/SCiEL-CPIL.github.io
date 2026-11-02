import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 46
    start_time = time.time()
    time_limit = 140.0
    
    # Schoolbook decomposition as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Check if a candidate solution is valid
    def is_valid(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check tensor equation
        # Residual for each i, k
        for i in range(n):
            for k in range(p):
                # Target: C_ik = sum_j A_ij B_jk
                # Coeff of A_ij B_jk in sum_r U V W is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # This must be 1 for all j
                # Coeff of A_ij B_j'k (j != j') must be 0
                for j in range(m):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
                for j_prime in range(m):
                    if j_prime == j: continue
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j_prime*p+k, r] * W[k*n+i, r]
                    if abs(val) > 1e-6:
                        return False
        return True

    # Try to find a solution with rank R
    def find_solution(rank):
        # Random initialization with small integers
        np.random.seed(42)
        U = np.random.randint(-1, 2, (n*m, rank))
        V = np.random.randint(-1, 2, (m*p, rank))
        W = np.random.randint(-1, 2, (p*n, rank))
        
        # Local search: try to fix residuals
        # This is a heuristic
        for _ in range(1000):
            if time.time() - start_time > time_limit:
                return None
            if is_valid(U, V, W):
                return {"U": U, "V": V, "W": W}
            # Randomly perturb one entry
            idx = np.random.randint(0, 3)
            if idx == 0:
                i, r = np.random.randint(0, n*m), np.random.randint(0, rank)
                U[i, r] += np.random.randint(-1, 2)
            elif idx == 1:
                i, r = np.random.randint(0, m*p), np.random.randint(0, rank)
                V[i, r] += np.random.randint(-1, 2)
            else:
                i, r = np.random.randint(0, p*n), np.random.randint(0, rank)
                W[i, r] += np.random.randint(-1, 2)
        return None

    # Try ranks from 53 down to 46
    for rank in range(53, 45, -1):
        if time.time() - start_time > time_limit:
            break
        sol = find_solution(rank)
        if sol is not None:
            return sol

    return get_schoolbook()
