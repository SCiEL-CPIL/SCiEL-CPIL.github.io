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
    
    # Precompute target tensor indices and values
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_exact(U, V, W):
        # U, V, W are numpy arrays
        rank = U.shape[1]
        T = np.zeros((n_m, m_p, p_n), dtype=int)
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Check target
        for a, b, c in target_indices:
            if T[a, b, c] != 1:
                return False
        # Check zeros
        # Create a mask of target positions
        mask = np.zeros((n_m, m_p, p_n), dtype=bool)
        for a, b, c in target_indices:
            mask[a, b, c] = True
        if not np.all(T[~mask] == 0):
            return False
        return True

    def count_errors(U, V, W):
        rank = U.shape[1]
        T = np.zeros((n_m, m_p, p_n), dtype=int)
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        errors = 0
        # Check target indices
        for a, b, c in target_indices:
            if T[a, b, c] != 1:
                errors += 1
        # Check zeros (sample for speed during search, full for final verification)
        # For local search, we need a consistent error metric.
        # Full check is 2500 elements. Rank 49.
        # 49 * 10 * 25 * 10 = 122,500 ops per evaluation.
        # If we do 1000 evaluations, that's 122M ops. Might be slow in Python/NumPy loop.
        # Let's optimize:
        # Only check a subset of zeros? No, we need to minimize total error.
        # But we can just check the target indices and assume zeros are handled if the structure is sparse?
        # Better: Just count errors on target indices + random zeros.
        
        # Actually, let's just use the target indices for the "loss" in local search,
        # and do a full verification only if target errors are 0.
        
        target_errors = 0
        for a, b, c in target_indices:
            if T[a, b, c] != 1:
                target_errors += 1
                
        # Sample zeros
        zero_errors = 0
        for _ in range(20):
            a = random.randint(0, n_m - 1)
            b = random.randint(0, m_p - 1)
            c = random.randint(0, p_n - 1)
            if (a, b, c) not in target_set:
                if T[a, b, c] != 0:
                    zero_errors += 1
                    
        return target_errors * 10 + zero_errors

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p
    
    # Search for rank 49
    target_rank = 49
    
    # Local search for rank 49
    # Initialize with random 0/1/-1
    # To improve chances, we can start from a "relaxed" version of schoolbook?
    # Schoolbook has 50 terms. We need 49.
    # We can try to merge two terms in schoolbook if possible?
    # Or just random init.
    
    # Optimization: Precompute row/col sums? No, just use NumPy einsum.
    
    num_restarts = 10
    steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
        V = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
        W = np.random.choice([-1, 0, 1], size=(p_n, target_rank))
        
        current_error = count_errors(U, V, W)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Pick a random column
            r = random.randint(0, target_rank - 1)
            
            # Pick a random matrix and entry to flip
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                idx = random.randint(0, n_m - 1)
                old_val = U[idx, r]
                U[idx, r] = -old_val if old_val != 0 else random.choice([-1, 1])
                # Simulate change
                # T changes by -2 * old_val * V[:,r] * W[:,r] (if flipping sign) or similar
                # For simplicity, recompute error.
                new_error = count_errors(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        # Verify exact
                        if verify_exact(U, V, W):
                            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = target_rank
                            return best
                else:
                    # Revert
                    U[idx, r] = old_val
            elif mat_choice == 1:
                idx = random.randint(0, m_p - 1)
                old_val = V[idx, r]
                V[idx, r] = -old_val if old_val != 0 else random.choice([-1, 1])
                new_error = count_errors(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        if verify_exact(U, V, W):
                            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = target_rank
                            return best
                else:
                    V[idx, r] = old_val
            else:
                idx = random.randint(0, p_n - 1)
                old_val = W[idx, r]
                W[idx, r] = -old_val if old_val != 0 else random.choice([-1, 1])
                new_error = count_errors(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error == 0:
                        if verify_exact(U, V, W):
                            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = target_rank
                            return best
                else:
                    W[idx, r] = old_val
                    
    return best
# EVOLVE-BLOCK-END
