import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Target tensor: T[i, j, k] = 1 for all i, j, k
    # In the decomposition sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want this sum to be 1 for all i, j, k.
    
    # Schoolbook decomposition
    U_school = np.zeros((n * m, target_rank))
    V_school = np.zeros((m * p, target_rank))
    W_school = np.zeros((p * n, target_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Function to compute the resulting tensor from U, V, W
    def compute_tensor(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Result T: (n, m, p)
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        R = U.shape[1]
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Reshape u to (n, m), v to (m, p), w to (p, n)
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            # The term is u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # This is an element-wise product of three matrices, but indices are coupled.
            # We can compute this efficiently using broadcasting or loops.
            # Since n, m, p are small, loops are fine.
            for i in range(n):
                for j in range(m):
                    u_val = u_mat[i, j]
                    if u_val == 0: continue
                    for k in range(p):
                        T[i, j, k] += u_val * v_mat[j, k] * w_mat[k, i]
        return T

    def error(U, V, W):
        T = compute_tensor(U, V, W)
        # Target is all ones
        return np.sum((T - 1.0) ** 2)
    
    start_time = time.time()
    timeout = 110 # 10s margin
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_error = 0.0
    best_rank = target_rank
    
    # Try to reduce rank by 1, 2, ...
    # We will use a local search on the entries.
    
    # To make it faster, we can precompute the target tensor as a flat array
    # and use a more efficient contraction.
    
    def fast_error(U, V, W):
        R = U.shape[1]
        # T_flat[i, j, k] index = (i * m + j) * p + k ? No, W index is k*n+i.
        # Let's stick to the 3D array for clarity, it's small.
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            # T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # This is equivalent to:
            # T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
            T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat, optimize=True)
        return np.sum((T - 1.0) ** 2)

    # Local search function
    def local_search(U, V, W, max_iters=1000):
        current_error = fast_error(U, V, W)
        R = U.shape[1]
        
        # We search over half-integers
        # Steps: change one entry in U, V, or W by +/- 0.5 or +/- 1
        
        for _ in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            # Randomly select a matrix and an entry
            mat_choice = np.random.choice(3)
            if mat_choice == 0:
                mat = U
                rows, cols = U.shape
            elif mat_choice == 1:
                mat = V
                rows, cols = V.shape
            else:
                mat = W
                rows, cols = W.shape
                
            i = np.random.randint(rows)
            j = np.random.randint(cols)
            
            old_val = mat[i, j]
            
            # Try some steps
            steps = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
            best_step = None
            best_err = current_error
            
            for step in steps:
                new_val = old_val + step
                # Optional: restrict to integers/half-integers
                # new_val = round(new_val * 2) / 2.0
                
                mat[i, j] = new_val
                new_err = fast_error(U, V, W)
                
                if new_err < best_err:
                    best_err = new_err
                    best_step = new_val
                mat[i, j] = old_val
                
            if best_step is not None:
                mat[i, j] = best_step
                current_error = best_err
                
                if current_error < 1e-9:
                    break
                    
        return U, V, W, current_error

    # Try to find a decomposition with lower rank
    # We start with the full rank and try to zero out columns?
    # No, we just optimize the existing columns.
    # To reduce rank, we need to find a decomposition with fewer columns.
    # This is harder.
    
    # For now, we just try to optimize the schoolbook decomposition.
    # It might find a non-trivial representation, but the rank is still 105.
    # To get a lower rank, we need to actually remove columns.
    
    # Strategy:
    # 1. Take the first R-1 columns of the schoolbook decomposition.
    # 2. Optimize them to fit the target tensor as closely as possible.
    # 3. If the error is 0, we have a valid decomposition of rank R-1.
    
    for new_rank in range(target_rank - 1, max(target_rank - 10, 0), -1):
        if time.time() - start_time > timeout:
            break
            
        U_sub = U_school[:, :new_rank].copy()
        V_sub = V_school[:, :new_rank].copy()
        W_sub = W_school[:, :new_rank].copy()
        
        # Optimize
        U_opt, V_opt, W_opt, err = local_search(U_sub, V_sub, W_sub, max_iters=500)
        
        if err < 1e-9:
            best_U = U_opt
            best_V = V_opt
            best_W = W_opt
            best_rank = new_rank
            break # Found a lower rank, stop
            
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
