import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _verify_fast(U, V, W, n, m, p):
    """
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Returns boolean if decomposition is exact.
    """
    r = U.shape[1]
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # Compute tensor T'
    # T'_{i, j, k} = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # einsum: 'ijc,jkc,kic->ijk'
    T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
    
    # Target tensor is all 1s
    target = np.ones((n, m, p))
    
    # Check exact match
    # Since entries are integers/half-integers, we can check for exact equality
    # or use a small tolerance if we were doing floating point optimization.
    # Here we assume U, V, W contain integers/half-integers.
    # np.allclose is safer if there's any floating point drift from previous steps, 
    # but if we stick to integers, exact is fine.
    return np.array_equal(T_prime, target)

def _local_search(U, V, W, n, m, p, max_steps=500, rng=None):
    """
    Perform local search to minimize error.
    Returns (U, V, W, success)
    """
    if rng is None:
        rng = np.random.RandomState()
        
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    target = np.ones((n, m, p))
    
    # Compute initial error
    T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
    error = np.sum((T_prime - target) ** 2)
    
    if error == 0:
        return U, V, W, True

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = error

    for step in range(max_steps):
        # Pick a random column
        c = rng.randint(r)
        
        # Pick a random matrix to update: 0=U, 1=V, 2=W
        mat_idx = rng.randint(3)
        
        # Pick a random row index for that matrix
        if mat_idx == 0:
            # U row: i*m + j
            idx = rng.randint(n * m)
            i, j = divmod(idx, m)
            # Update U[i, j, c]
            old_val = U_3d[i, j, c]
            # Try adding/subtracting 1 or flipping
            # Since we want integers/half-integers, let's just try +-1 steps
            # Or try setting to 0, 1, -1
            candidates = [old_val + 1, old_val - 1, 0, 1, -1]
            # Filter duplicates
            candidates = list(set(candidates))
            best_new_val = old_val
            best_new_error = error
            for new_val in candidates:
                U_3d[i, j, c] = new_val
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
                new_error = np.sum((T_prime - target) ** 2)
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_new_val = new_val
            U_3d[i, j, c] = best_new_val
            error = best_new_error
            
        elif mat_idx == 1:
            # V row: j*p + k
            idx = rng.randint(m * p)
            j, k = divmod(idx, p)
            old_val = V_3d[j, k, c]
            candidates = [old_val + 1, old_val - 1, 0, 1, -1]
            candidates = list(set(candidates))
            best_new_val = old_val
            best_new_error = error
            for new_val in candidates:
                V_3d[j, k, c] = new_val
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
                new_error = np.sum((T_prime - target) ** 2)
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_new_val = new_val
                V_3d[j, k, c] = old_val # Restore for next candidate
            V_3d[j, k, c] = best_new_val
            error = best_new_error
            
        else:
            # W row: k*n + i
            idx = rng.randint(p * n)
            k, i = divmod(idx, n)
            old_val = W_3d[k, i, c]
            candidates = [old_val + 1, old_val - 1, 0, 1, -1]
            candidates = list(set(candidates))
            best_new_val = old_val
            best_new_error = error
            for new_val in candidates:
                W_3d[k, i, c] = new_val
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
                new_error = np.sum((T_prime - target) ** 2)
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_new_val = new_val
                W_3d[k, i, c] = old_val
            W_3d[k, i, c] = best_new_val
            error = best_new_error

        if error == 0:
            return U_3d.reshape(n*m, r), V_3d.reshape(m*p, r), W_3d.reshape(p*n, r), True

        if error < best_error:
            best_error = error
            best_U = U_3d.copy()
            best_V = V_3d.copy()
            best_W = W_3d.copy()
            
        # Simulated annealing / random restart within search
        if step % 50 == 0 and error > 0:
             # Occasionally accept a worse move to escape local minima?
             # For now, strict hill climbing.
             pass

    return best_U, best_V, best_W, False

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115s safety margin
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try ranks from 60 down to 25
    # We prioritize lower ranks but ensure we have time.
    # If we find a rank 40 solution, we can stop or continue to 30?
    # The score is seed/value. Lower value is better.
    # So we want the MINIMUM rank.
    # We should search from low to high? No, if we start from low, we might waste time.
    # Start from a reasonable guess (e.g. 50) and go down.
    # If we fail at 50, try 45, etc.
    
    # Actually, let's try a range of ranks.
    # For 3x5x5, let's try 45, 40, 35, 30, 25.
    ranks_to_try = [50, 45, 40, 35, 30, 25]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 200
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(restart * 1000 + r)
            
            # Random sparse initialization
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Density ~ 0.1
            # Each column has ~1-2 non-zeros
            for c in range(r):
                # U
                if rng.rand() < 0.8:
                    idx = rng.randint(n*m)
                    U[idx, c] = rng.choice([-1, 1])
                # V
                if rng.rand() < 0.8:
                    idx = rng.randint(m*p)
                    V[idx, c] = rng.choice([-1, 1])
                # W
                if rng.rand() < 0.8:
                    idx = rng.randint(p*n)
                    W[idx, c] = rng.choice([-1, 1])
            
            # Check if lucky
            if _verify_fast(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
            
            # Local search
            U_ls, V_ls, W_ls, success = _local_search(U, V, W, n, m, p, max_steps=200, rng=rng)
            if success:
                best_U, best_V, best_W = U_ls, V_ls, W_ls
                best_rank = r
                found = True
                break
                
        if found:
            # Found a solution at rank r. 
            # Should we try lower ranks?
            # Yes, because lower rank = better score.
            # But if we found 50, 45 might be possible.
            # We continue the loop to lower ranks.
            pass

    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Ensure integer/half-integer format (they are already floats/ints from numpy)
    # The evaluator expects lists of numbers.
    return {"U": U_out, "V": V_out, "W": W_out}
