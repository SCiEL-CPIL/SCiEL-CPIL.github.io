import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Precompute target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_indices = np.array(target_indices)
    target_set = set(map(tuple, target_indices))
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n_m, rank), dtype=int)
        V = np.zeros((m_p, rank), dtype=int)
        W = np.zeros((p_n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def fast_error(U, V, W, zero_samples):
        """
        Calculate error using dot products.
        U, V, W are numpy arrays.
        zero_samples is a list of (a,b,c) tuples.
        """
        error = 0
        # Check targets
        for a, b, c in target_indices:
            val = np.dot(U[a], V[b] * W[c])
            if val != 1:
                error += abs(val - 1)
                
        # Check zeros
        for a, b, c in zero_samples:
            val = np.dot(U[a], V[b] * W[c])
            if val != 0:
                error += abs(val)
                
        return error

    def verify_exact(U, V, W):
        rank = U.shape[1]
        # Full check
        T = np.zeros((n_m, m_p, p_n), dtype=int)
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        for a, b, c in target_indices:
            if T[a, b, c] != 1:
                return False
        mask = np.zeros((n_m, m_p, p_n), dtype=bool)
        for a, b, c in target_indices:
            mask[a, b, c] = True
        if not np.all(T[~mask] == 0):
            return False
        return True

    best = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = best['U'], best['V'], best['W']
    
    target_rank = 49
    
    # Generate fixed set of zero samples for consistent error metric
    zero_samples = []
    while len(zero_samples) < 50:
        a = random.randint(0, n_m - 1)
        b = random.randint(0, m_p - 1)
        c = random.randint(0, p_n - 1)
        if (a, b, c) not in target_set:
            zero_samples.append((a, b, c))
            
    # Strategy: Start from schoolbook, remove one term, and try to fix
    num_restarts = 5
    steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize by taking schoolbook and zeroing out a random column
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        
        # Remove one term
        remove_col = random.randint(0, best_rank - 1)
        
        # Re-index to keep 49 columns
        cols_to_keep = [c for c in range(best_rank) if c != remove_col]
        U = U[:, cols_to_keep]
        V = V[:, cols_to_keep]
        W = W[:, cols_to_keep]
        
        current_error = fast_error(U, V, W, zero_samples)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            r = random.randint(0, target_rank - 1)
            mat_choice = random.randint(0, 2)
            
            if mat_choice == 0:
                idx = random.randint(0, n_m - 1)
                old_val = U[idx, r]
                possible_vals = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(possible_vals)
                U[idx, r] = new_val
                new_error = fast_error(U, V, W, zero_samples)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        if verify_exact(U, V, W):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    U[idx, r] = old_val
            elif mat_choice == 1:
                idx = random.randint(0, m_p - 1)
                old_val = V[idx, r]
                possible_vals = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(possible_vals)
                V[idx, r] = new_val
                new_error = fast_error(U, V, W, zero_samples)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        if verify_exact(U, V, W):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    V[idx, r] = old_val
            else:
                idx = random.randint(0, p_n - 1)
                old_val = W[idx, r]
                possible_vals = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(possible_vals)
                W[idx, r] = new_val
                new_error = fast_error(U, V, W, zero_samples)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        if verify_exact(U, V, W):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    W[idx, r] = old_val
                    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
