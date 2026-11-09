import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin
    
    n = 20
    
    # Precompute subset incidence matrix X
    # X[mask, v] = 1 if v is in mask
    masks = np.arange(1 << n, dtype=np.int64)
    # Vectorized creation of X
    # (1 << n) x n matrix
    X = ((masks[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    
    # Identify masks with size >= 6
    # popcount of masks
    popcounts = np.array([m.bit_count() for m in masks], dtype=np.uint8)
    valid_mask_indices = np.where(popcounts >= 6)[0]
    
    def get_violations(D):
        # D is (1<<n) x n
        # We only care about valid_mask_indices
        # For each valid mask, check if degrees are all equal
        # Degrees are D[mask, :]
        # But we only consider vertices in the mask.
        # We can use the trick: set invalid vertices to a large value.
        # However, X is already computed.
        # valid_D = np.where(X == 1, D, 100)
        # mins = valid_D.min(axis=1)
        # maxs = valid_D.max(axis=1)
        # is_reg = (mins == maxs)
        # return np.sum(is_reg[valid_mask_indices])
        
        # To speed up, we can precompute the "invalid" mask?
        # Actually, just doing the where/min/max on the subset of valid masks is fine.
        
        D_valid = D[valid_mask_indices]
        X_valid = X[valid_mask_indices]
        
        # Replace 0s in X_valid with 100? No, we want to ignore them.
        # If we use np.where(X_valid == 1, D_valid, 100), then min/max will ignore the 100s if there's any 1.
        # Since popcount >= 6, there is at least one 1.
        
        masked_D = np.where(X_valid == 1, D_valid, 100)
        mins = masked_D.min(axis=1)
        maxs = masked_D.max(axis=1)
        
        is_reg = (mins == maxs)
        return np.sum(is_reg)

    # Initialize random graph
    A = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                A[i, j] = 1
                A[j, i] = 1
                
    D = X @ A.T
    current_violations = get_violations(D)
    
    best_A = A.copy()
    best_violations = current_violations
    
    if best_violations == 0:
        return _format_result(n, best_A)
        
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        if current_violations == 0:
            break
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        has_edge = A[u, v] == 1
        if has_edge:
            delta = -1
        else:
            delta = 1
            
        # Update D
        # D[:, u] += X[:, v] * delta
        # D[:, v] += X[:, u] * delta
        # Note: X is uint8, D is int. Need to cast or handle types.
        # D is result of uint8 @ uint8.T, so it's int.
        
        # Save old values to revert if needed?
        # Or just compute new_violations and revert if worse.
        
        D[:, u] += X[:, v] * delta
        D[:, v] += X[:, u] * delta
        
        A[u, v] = 1 - A[u, v]
        A[v, u] = 1 - A[v, u]
        
        new_violations = get_violations(D)
        
        if new_violations < current_violations:
            current_violations = new_violations
            best_A = A.copy()
            best_violations = current_violations
            no_improve = 0
        else:
            # Revert
            D[:, u] -= X[:, v] * delta
            D[:, v] -= X[:, u] * delta
            
            A[u, v] = 1 - A[u, v]
            A[v, u] = 1 - A[v, u]
            
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart
                A = np.zeros((n, n), dtype=np.uint8)
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            A[i, j] = 1
                            A[j, i] = 1
                            
                D = X @ A.T
                current_violations = get_violations(D)
                best_A = A.copy()
                best_violations = current_violations
                no_improve = 0
                if best_violations == 0:
                    break

    return _format_result(n, best_A)

def _format_result(n, A):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
