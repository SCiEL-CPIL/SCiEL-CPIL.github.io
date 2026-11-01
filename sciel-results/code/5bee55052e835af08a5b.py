import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s
    
    n, m, p = 2, 2, 8
    
    # First, get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()
    
    # Try to find a lower-rank decomposition via random search
    # We prioritize lower ranks first
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 5:  # Need at least 5s to try and verify
            break
        
        # Allocate time budget for this rank
        # Give more time to higher ranks (closer to 28) as they are more likely
        # But we want the lowest rank, so we try 24 first.
        # If 24 is impossible, we waste time.
        # Heuristic: spend 20% of time on 24, 20% on 25, 20% on 26, 40% on 27?
        # Or just equal split?
        # Let's give 10s to 24, 10s to 25, 10s to 26, and rest to 27.
        budget_map = {24: 10, 25: 10, 26: 10, 27: 20}
        budget = min(budget_map.get(target_rank, 10), remaining - 5)
        
        if budget <= 1:
            continue

        result = _search_low_rank(target_rank, budget)
        if result is not None:
            return result
    
    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using random restarts and local refinement."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Non-zero indices of the tensor
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))
    
    nonzero_set = set(nonzero_indices)
    
    # Pre-compute zero indices for faster checking?
    # There are 4*16*16 = 1024 total. 32 non-zero. 992 zero.
    zero_indices = []
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                if (a, b, c) not in nonzero_set:
                    zero_indices.append((a, b, c))
    
    def check_solution(U, V, W):
        # Compute T_hat
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        T_hat = np.einsum('abr,cr->abc', UV, W)
        
        # Check non-zero
        for a, b, c in nonzero_indices:
            if abs(T_hat[a, b, c] - 1) > 1e-6:
                return False
        
        # Check zero (sample if too slow, but 992 is small)
        for a, b, c in zero_indices:
            if abs(T_hat[a, b, c]) > 1e-6:
                return False
        return True

    # Pre-compute target tensor for error calculation
    target_tensor = np.zeros((num_a, num_b, num_c), dtype=float)
    for a, b, c in nonzero_indices:
        target_tensor[a, b, c] = 1.0
    
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.default_rng(seed=restart)
        
        # Initialize with small integers
        U = rng.integers(-3, 4, size=(num_a, target_rank)).astype(float)
        V = rng.integers(-3, 4, size=(num_b, target_rank)).astype(float)
        W = rng.integers(-3, 4, size=(num_c, target_rank)).astype(float)
        
        # Local Refinement: Coordinate Descent with Incremental Updates
        for step in range(500):
            if time.time() - start > time_budget:
                break
            
            # Compute residual and error
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            T_hat = np.einsum('abr,cr->abc', UV, W)
            
            # Calculate error using vectorized operations
            diff = T_hat - target_tensor
            error = np.sum(diff * diff)
            
            # If error is small, check exactness
            if error < 1e-3:
                if check_solution(U, V, W):
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_solution(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Try a random coordinate update
            mat_choice = rng.integers(0, 3)
            if mat_choice == 0:
                mat = U
                rows = num_a
            elif mat_choice == 1:
                mat = V
                rows = num_b
            else:
                mat = W
                rows = num_c
            
            r_idx = rng.integers(0, rows)
            c_idx = rng.integers(0, target_rank)
            
            old_val = mat[r_idx, c_idx]
            
            best_delta = None
            best_error = error
            
            # Try a few new values
            for delta in [-2, -1, -0.5, 0.5, 1, 2]:
                new_val = old_val + delta
                if new_val == old_val:
                    continue
                
                # Compute change in T_hat incrementally
                if mat_choice == 0:  # U changed
                    dT = np.einsum('b,c->abc', V[:, c_idx], W[:, c_idx]) * delta
                elif mat_choice == 1:  # V changed
                    dT = np.einsum('a,c->abc', U[:, c_idx], W[:, c_idx]) * delta
                else:  # W changed
                    dT = np.einsum('a,b->abc', U[:, c_idx], V[:, c_idx]) * delta
                
                # Incremental error update:
                # new_error = sum((T_hat + dT - target)^2)
                # = sum((T_hat - target)^2) + 2*sum((T_hat - target)*dT) + sum(dT^2)
                # = error + 2*sum(diff*dT) + sum(dT*dT)
                term1 = 2.0 * np.sum(diff * dT)
                term2 = np.sum(dT * dT)
                new_error = error + term1 + term2
                
                if new_error < best_error:
                    best_error = new_error
                    best_delta = delta
                    best_diff = diff + dT
            
            if best_delta is not None:
                # Accept the update
                mat[r_idx, c_idx] += best_delta
                diff = best_diff
                error = best_error
        
    return None
