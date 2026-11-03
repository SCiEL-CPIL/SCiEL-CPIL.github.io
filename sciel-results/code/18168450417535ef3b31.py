import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: find rank-minimal exact tensor decomposition
    # Lader (1976) proved rank 33 is achievable for 3x3x4.
    # We aim to find a rank-33 decomposition.
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Flatten T for efficient computation
    T_flat = T.flatten()
    
    # Precompute indices for the 36 non-zero entries
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                nz_indices.append((a, b, c))
    
    def compute_error(U, V, W):
        """Compute ||X - T||_F^2 where X = sum_c outer(U[:,c], V[:,c], W[:,c])"""
        r = U.shape[1]
        # For each of the 36 non-zero positions, compute the sum
        # X[a,b,c] = sum_c U[a,c]*V[b,c]*W[c_idx,c]
        # We only need to check the 36 positions that should be 1,
        # and all other positions should be 0.
        
        # Efficient: compute all 9*12*12 = 1296 entries
        # But that's expensive. Instead, use einsum for the full tensor.
        X = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.sum((X - T)**2)
    
    def is_valid(U, V, W, tol=1e-9):
        """Check if U,V,W form a valid decomposition"""
        r = U.shape[1]
        X = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(X, T, atol=tol)
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Strategy: Use a structured approach based on known rank-33 constructions
    # For 3x3x4, we can try to adapt the 3x3x3 Strassen-like approach
    # or use a direct search with good initialization.
    
    # Approach 1: Try to find rank 33 using random search with local optimization
    # The key insight: entries should be from {-1, -0.5, 0, 0.5, 1}
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Precompute the target values at non-zero positions
    target_vals = np.ones(len(nz_indices))
    
    # For faster error computation, precompute which positions are non-zero
    # and use a sparse check
    
    def fast_error(U, V, W):
        """Fast error computation using only the 36 target positions + zero positions"""
        r = U.shape[1]
        # Compute X at all 1296 positions using einsum (fast with numpy)
        X = np.einsum('ac,bc,dc->abd', U, V, W)
        # Error = sum over all positions of (X - T)^2
        # Since T is mostly zeros, this is sum(X^2) - 2*sum(X*T) + sum(T^2)
        # sum(T^2) = 36
        # sum(X*T) = sum over nz of X[a,b,c]
        err = np.sum(X**2) - 2.0 * np.sum(X * T) + 36.0
        return err
    
    def fast_is_valid(U, V, W, tol=1e-9):
        X = np.einsum('ac,bc,dc->abd', U, V, W)
        # Check all entries
        return np.allclose(X, T, atol=tol)
    
    # Local search: perturb one entry at a time to reduce error
    def local_search(U, V, W, max_iter=5000, time_budget=20.0):
        """Local search to minimize error"""
        current_err = fast_error(U, V, W)
        
        if current_err < 1e-9:
            return U, W, current_err, True
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        ns = time.time()
        for it in range(max_iter):
            if time.time() - ns > time_budget:
                break
            
            # Try to improve by changing one entry
            # Randomly pick which matrix and which entry
            mat_choice = random.random()
            if mat_choice < 0.333:
                mat = 'U'
                arr = U
            elif mat_choice < 0.667:
                mat = 'V'
                arr = V
            else:
                mat = 'W'
                arr = W
            
            rows, cols = arr.shape
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            
            old_val = arr[row, col]
            
            # Try all possible values
            best_val = old_val
            best_err = current_err
            
            for val in values:
                if val == old_val:
                    continue
                arr[row, col] = val
                new_err = fast_error(U, V, W)
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    best_val = val
                arr[row, col] = old_val
            
            if best_val != old_val:
                arr[row, col] = best_val
                current_err = best_err
                if current_err < 1e-9:
                    if mat == 'U':
                        return U, V, W, current_err, True
                    elif mat == 'V':
                        return U, V, W, current_err, True
                    else:
                        return U, V, W, current_err, True
            else:
                # No improvement, try a random jump occasionally
                if random.random() < 0.1:
                    arr[row, col] = random.choice(values)
                    current_err = fast_error(U, V, W)
        
        return U, V, W, current_err, False
    
    # Try rank 33
    r_target = 33
    found = False
    
    # Multiple random restarts
    n_restarts = 50
    restart_time_budget = (time_limit - 10) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 15:
            break
        
        # Initialize with random values
        U = np.random.choice(values, size=(n*m, r_target))
        V = np.random.choice(values, size=(m*p, r_target))
        W = np.random.choice(values, size=(p*n, r_target))
        
        # Check if already valid
        if fast_is_valid(U, V, W):
            found = True
            break
        
        # Local search
        U, V, W, err, success = local_search(U, V, W, max_iter=3000, time_budget=restart_time_budget)
        
        if success:
            found = True
            break
    
    if found:
        # Final verification
        if fast_is_valid(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 34
    if time.time() - start_time < time_limit - 30:
        r_target = 34
        for restart in range(20):
            if time.time() - start_time > time_limit - 20:
                break
            
            U = np.random.choice(values, size=(n*m, r_target))
            V = np.random.choice(values, size=(m*p, r_target))
            W = np.random.choice(values, size=(p*n, r_target))
            
            if fast_is_valid(U, V, W):
                found = True
                break
            
            U, V, W, err, success = local_search(U, V, W, max_iter=2000, time_budget=5.0)
            
            if success:
                found = True
                break
        
        if found and fast_is_valid(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 36
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
