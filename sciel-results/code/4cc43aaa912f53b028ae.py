# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # Precompute diagonal of A
    a_diag = np.diag(a)
    
    # Fast 2-Opt using vectorized delta calculation
    # We will iterate until no improvement or time limit
    while (time.time() - start_time) < deadline:
        # Compute B_perm and M
        B_perm = b[np.ix_(p, p)]
        M = a @ B_perm
        S3 = np.sum(a * B_perm, axis=1)
        B_diag = np.diag(B_perm)
        
        improved = False
        # Iterate over all pairs to find best improvement
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            # Vectorized delta calculation for all j > i
            # Base = M[i, j] - S3[i] - S3[j] + M[j, i]
            Base_vec = M[i, :] - S3[i] - S3 - M[:, i]
            
            # Ti = (a[i, i] - a[i, j]) * (B_perm[j, i] - B_perm[i, i])
            # Note: B_perm is symmetric, so B_perm[j, i] = B_perm[i, j]
            Ti_vec = (a[i, i] - a[i, :]) * (B_perm[i, :] - B_perm[i, i])
            
            # Tj = (a[i, j] - a[j, j]) * (B_perm[j, j] - B_perm[i, j])
            Tj_vec = (a[i, :] - a_diag) * (B_diag - B_perm[i, :])
            
            # DiagCorr = (a[j, j] - a[i, i]) * (B_perm[i, i] - B_perm[j, j])
            DiagCorr_vec = (a_diag - a[i, i]) * (B_perm[i, i] - B_diag)
            
            Deltas = Base_vec - Ti_vec - Tj_vec + DiagCorr_vec
            
            # Only consider j > i
            if i < n - 1:
                valid_indices = np.arange(i + 1, n)
                valid_deltas = Deltas[valid_indices]
                
                min_val = np.min(valid_deltas)
                if min_val < best_delta:
                    best_delta = min_val
                    min_idx_in_valid = np.argmin(valid_deltas)
                    best_i = i
                    best_j = valid_indices[min_idx_in_valid]
        
        if best_delta < 0:
            # Apply swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
            # Check time
            if time.time() - start_time > deadline:
                break
        else:
            # No improvement found in this pass
            break
            
    # 2. Simulated Annealing with remaining time
    if time.time() - start_time < deadline - 5.0: # Leave 5s buffer
        T = 5000.0 
        T_end = 1.0
        alpha = 0.995
        
        # Use current best as start for SA
        p = best_p.copy()
        current_cost = best_cost
        
        while T > T_end and (time.time() - start_time) < deadline:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            # Calculate delta for swap i, j
            # We can reuse the logic or just do a direct calc since it's O(n)
            # Direct calc is safer for SA where we don't need full pass
            dA = a[i, :] - a[j, :]
            dB = b[p[j], p] - b[p[i], p]
            dB[i] = 0
            dB[j] = 0
            delta = np.dot(dA, dB)
            
            if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
            T *= alpha
            
            if time.time() - start_time > deadline:
                break
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
