"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fast verification using tensor equation
    def verify_fast(U, V, W, num_tests=10):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        for _ in range(num_tests):
            A = np.random.randint(-3, 4, (n, m)).astype(float)
            B = np.random.randint(-3, 4, (m, p)).astype(float)
            C_expected = A @ B
            
            # S_r = sum_{i,j} U[i*m+j, r] * A[i,j]
            A_flat = A.flatten()
            S = U_arr.T @ A_flat  # Shape (R,)
            B_flat = B.flatten()
            T = V_arr.T @ B_flat  # Shape (R,)
            
            # C[i,k] = sum_r W[k*n+i, r] * S[r] * T[r]
            ST = S * T  # Shape (R,)
            # Reshape W to (p, n, R) and compute
            W_reshaped = W_arr.reshape(p, n, R)
            C_computed = np.einsum('pnr,r->np', W_reshaped, ST)
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Local optimization: try to zero out entries while maintaining validity
    def local_optimize(U, V, W, max_iter=2000):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try zeroing random entries
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row, col = random.randint(0, U_arr.shape[0]-1), random.randint(0, R-1)
                    if U_arr[row, col] == 0:
                        continue
                    old_val = U_arr[row, col]
                    U_arr[row, col] = 0
                    if verify_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), num_tests=3):
                        best_U = U_arr.copy()
                        improved = True
                    else:
                        U_arr[row, col] = old_val
                elif mat == 'V':
                    row, col = random.randint(0, V_arr.shape[0]-1), random.randint(0, R-1)
                    if V_arr[row, col] == 0:
                        continue
                    old_val = V_arr[row, col]
                    V_arr[row, col] = 0
                    if verify_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), num_tests=3):
                        best_V = V_arr.copy()
                        improved = True
                    else:
                        V_arr[row, col] = old_val
                else:
                    row, col = random.randint(0, W_arr.shape[0]-1), random.randint(0, R-1)
                    if W_arr[row, col] == 0:
                        continue
                    old_val = W_arr[row, col]
                    W_arr[row, col] = 0
                    if verify_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), num_tests=3):
                        best_W = W_arr.copy()
                        improved = True
                    else:
                        W_arr[row, col] = old_val
            
            if not improved:
                break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist()

    # Structured search: start from schoolbook and try to merge terms
    def structured_search(target_rank, num_trials=50):
        if target_rank >= n * m * p:
            return get_schoolbook()
        
        # Start with schoolbook and try to zero out random columns
        base = get_schoolbook()
        U = np.array(base['U'], dtype=float)
        V = np.array(base['V'], dtype=float)
        W = np.array(base['W'], dtype=float)
        
        full_rank = n * m * p
        
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            
            # Randomly select columns to try to eliminate
            U_trial = U.copy()
            V_trial = V.copy()
            W_trial = W.copy()
            
            # Try to zero out (full_rank - target_rank) columns by merging
            cols_to_try = list(range(full_rank))
            random.shuffle(cols_to_try)
            
            eliminated = 0
            for col in cols_to_try:
                if eliminated >= (full_rank - target_rank):
                    break
                if time.time() > deadline - 10.0:
                    break
                
                # Try to zero this column by redistributing its contribution
                # This is complex; instead just try random perturbations
                pass
            
            # Alternative: random restart with structured initialization
            # Use a subset of the schoolbook terms with random signs
            U_rand = np.zeros((n * m, target_rank), dtype=float)
            V_rand = np.zeros((m * p, target_rank), dtype=float)
            W_rand = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with random subset of schoolbook structure
            used_cols = random.sample(range(full_rank), min(target_rank, full_rank))
            for new_col, old_col in enumerate(used_cols):
                sign = random.choice([-1, 1])
                U_rand[:, new_col] = U[:, old_col] * sign
                V_rand[:, new_col] = V[:, old_col] * sign
                W_rand[:, new_col] = W[:, old_col] * sign
            
            if verify_fast(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), num_tests=5):
                # Local optimize
                U_opt, V_opt, W_opt = local_optimize(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), max_iter=500)
                return {"U": U_opt, "V": V_opt, "W": W_opt}
        
        return None

    # Pure random search with better initialization
    def random_search(target_rank, num_trials=100):
        for _ in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            
            # Random entries from {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_fast(U, V, W, num_tests=5):
                U_opt, V_opt, W_opt = local_optimize(U, V, W, max_iter=300)
                return {"U": U_opt, "V": V_opt, "W": W_opt}
        
        return None

    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s safety margin
    
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Try decreasing ranks
    for target_rank in range(44, 40, -1):
        if time.time() > deadline - 20.0:
            break
        
        # Try structured search first
        result = structured_search(target_rank, num_trials=30)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue trying lower ranks
            
        if time.time() > deadline - 20.0:
            break
        
        # Try random search
        result = random_search(target_rank, num_trials=50)
        if result is not None:
            best_result = result
            best_rank = target_rank
    
    # Final verification of best result
    if verify_fast(best_result['U'], best_result['V'], best_result['W'], num_tests=20):
        return best_result
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
