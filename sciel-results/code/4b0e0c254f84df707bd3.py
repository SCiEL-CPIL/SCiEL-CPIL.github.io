import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=2, m=3, p=3
    # U: 6xR, V: 9xR, W: 6xR
    
    # Target Tensor T[A, B, C]
    # A = i*m + j
    # B = j*p + k
    # C = k*n + i
    # T is 1 if (A,B,C) is in the support, 0 otherwise.
    
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                support.append((A, B, C))
                
    # We want to find U, V, W such that sum_r U[A,r]V[B,r]W[C,r] = T[A,B,C]
    
    # Try ranks from 11 to 17
    start_time = time.time()
    time_limit = 110
    
    # Helper to check validity
    def check(U, V, W, R):
        # U, V, W are numpy arrays of shape (rows, R)
        # Calculate T_reconstructed
        # T_reconstructed[A, B, C] = sum_r U[A,r] V[B,r] W[C,r]
        # This is a tensor contraction.
        # We can do this efficiently using einsum or loops.
        # Given small sizes, loops might be fine, but einsum is better.
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if T_recon matches target T.
        
        # Construct target T as a sparse representation or dense array
        # Dense is 6x9x6 = 324 floats. Very small.
        T_target = np.zeros((n*m, m*p, p*n))
        for A, B, C in support:
            T_target[A, B, C] = 1.0
            
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check exact match (with tolerance for floating point, but we use integers/half-integers)
        # Since we use integers/half-integers, the result should be exact if the math is exact.
        # But floating point might introduce tiny errors.
        # We check if max abs diff is < 1e-9
        if np.allclose(T_recon, T_target, atol=1e-9):
            return True
        return False

    # Helper to calculate error
    def calc_error(U, V, W, T_target):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_recon - T_target)**2)

    # Precompute T_target
    T_target = np.zeros((n*m, m*p, p*n))
    for A, B, C in support:
        T_target[A, B, C] = 1.0

    # Local Search
    def local_search(R, max_iterations=50000, time_limit=110):
        # Initialize with random integers in {-2, -1, 0, 1, 2}
        # Or start from schoolbook and remove columns?
        # Let's try random init first.
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # We can also try to start from a known lower rank if we had one, but we don't.
        # Let's use a "warm start" from a higher rank if we found one?
        # For now, random restarts.
        
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
            
            # Scale? Maybe random integers are too sparse.
            # Let's try Gaussian integers? No, task says integer/half-integer.
            
            err = calc_error(U, V, W, T_target)
            if err < 1e-9:
                return U, V, W, R
            
            # Local search
            for it in range(max_iterations):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random variable to change
                # There are R*(n*m + m*p + p*n) variables.
                total_vars = R * (n*m + m*p + p*n)
                var_idx = np.random.randint(0, total_vars)
                
                # Identify which matrix and index
                if var_idx < R * n * m:
                    # U
                    r = var_idx % R
                    row = var_idx // R
                    # Try changing U[row, r] by +/- 0.5 or +/- 1
                    delta = np.random.choice([0.5, -0.5, 1, -1])
                    old_val = U[row, r]
                    U[row, r] = old_val + delta
                    new_err = calc_error(U, V, W, T_target)
                    if new_err < err:
                        err = new_err
                        if err < 1e-9:
                            return U, V, W, R
                    else:
                        U[row, r] = old_val
                        
                elif var_idx < R * (n*m + m*p):
                    # V
                    idx_in_v = var_idx - R * n * m
                    r = idx_in_v % R
                    row = idx_in_v // R
                    delta = np.random.choice([0.5, -0.5, 1, -1])
                    old_val = V[row, r]
                    V[row, r] = old_val + delta
                    new_err = calc_error(U, V, W, T_target)
                    if new_err < err:
                        err = new_err
                        if err < 1e-9:
                            return U, V, W, R
                    else:
                        V[row, r] = old_val
                else:
                    # W
                    idx_in_w = var_idx - R * (n*m + m*p)
                    r = idx_in_w % R
                    row = idx_in_w // R
                    delta = np.random.choice([0.5, -0.5, 1, -1])
                    old_val = W[row, r]
                    W[row, r] = old_val + delta
                    new_err = calc_error(U, V, W, T_target)
                    if new_err < err:
                        err = new_err
                        if err < 1e-9:
                            return U, V, W, R
                    else:
                        W[row, r] = old_val
                        
            # If we didn't find a solution in this restart, try another
            pass
            
        return None

    # Try ranks
    for R in range(11, 18):
        if time.time() - start_time > time_limit:
            break
        res = local_search(R)
        if res is not None:
            U, V, W, r_found = res
            # Convert to list of lists for output
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                U[A, col] = 1
                V[B, col] = 1
                W[C, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
