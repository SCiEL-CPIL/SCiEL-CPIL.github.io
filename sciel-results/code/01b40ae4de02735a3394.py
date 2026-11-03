import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Schoolbook decomposition (Rank 60)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Efficient tensor error computation using vectorized operations
    # T_rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T_rec[i, j, k] = 1 for all i, j, k
    
    # Precompute index mappings
    # For each (i, j, k), we need:
    #   U_idx = i*m + j
    #   V_idx = j*p + k  
    #   W_idx = k*n + i
    
    # Build lookup tables for efficient computation
    U_indices = np.zeros((n, m, p), dtype=int)
    V_indices = np.zeros((n, m, p), dtype=int)
    W_indices = np.zeros((n, m, p), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_indices[i, j, k] = i * m + j
                V_indices[i, j, k] = j * p + k
                W_indices[i, j, k] = k * n + i

    def compute_error_fast(U, V, W, R):
        """Compute sum of squared errors using vectorized operations"""
        # For each r, compute contribution to all (i,j,k)
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient approach: for each r, outer product structure
        # U_col = U[:, r] has shape (n*m,)
        # V_col = V[:, r] has shape (m*p,)
        # W_col = W[:, r] has shape (p*n,)
        
        # Reshape to 3D:
        # U_3d[i, j, r] = U[i*m+j, r]
        # V_3d[j, k, r] = V[j*p+k, r]
        # W_3d[k, i, r] = W[k*n+i, r]
        
        # T_rec[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        
        # Vectorized: 
        # U_3d shape: (n, m, R)
        # V_3d shape: (m, p, R)
        # W_3d shape: (p, n, R)
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # For each r: sum over r of U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # This is a 3-way contraction
        
        # Use einsum: sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # Result shape: (n, m, p)
        
        # First contract U and V over r: (n, m, p)
        # Actually: sum_r U_3d[i,j,r] * V_3d[j,k,r] gives (n, m, p)
        # Then multiply by W_3d[k,i,r]... but we need to sum over r
        
        # Better: use einsum directly
        # T_rec[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        
        # einsum: 'ijr,jkr,kir->ijk'
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        
        # Error = sum (T_rec - 1)^2
        err = np.sum((T_rec - 1.0) ** 2)
        return err, T_rec

    def try_rank(R, max_iters=2000, init_type='random'):
        if time.time() > deadline - 5:
            return None
        
        if init_type == 'random':
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
        elif init_type == 'sparse':
            # Sparse initialization: mostly zeros with some 1s and -1s
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            W = np.zeros((p*n, R), dtype=int)
            density = 0.3
            for idx in range(n*m):
                for r in range(R):
                    if random.random() < density:
                        U[idx, r] = random.choice([-1, 1])
            for idx in range(m*p):
                for r in range(R):
                    if random.random() < density:
                        V[idx, r] = random.choice([-1, 1])
            for idx in range(p*n):
                for r in range(R):
                    if random.random() < density:
                        W[idx, r] = random.choice([-1, 1])
        elif init_type == 'structured':
            # Try to build from schoolbook and remove columns
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            W = np.zeros((p*n, R), dtype=int)
            # Start with first R of schoolbook
            for col in range(min(R, rank_sb)):
                i = col // (m*p)
                rem = col % (m*p)
                j = rem // p
                k = rem % p
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
            # Randomly perturb
            for _ in range(R * 2):
                mat = random.choice(3)
                if mat == 0:
                    idx = random.randint(0, n*m-1)
                    r = random.randint(0, R-1)
                    U[idx, r] = random.choice([-1, 0, 1])
                elif mat == 1:
                    idx = random.randint(0, m*p-1)
                    r = random.randint(0, R-1)
                    V[idx, r] = random.choice([-1, 0, 1])
                else:
                    idx = random.randint(0, p*n-1)
                    r = random.randint(0, R-1)
                    W[idx, r] = random.choice([-1, 0, 1])
        else:
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
        
        current_err, T_rec = compute_error_fast(U, V, W, R)
        
        if current_err == 0:
            return U, V, W
        
        # Local search with vectorized error computation
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry to modify
            mat_choice = np.random.choice(3)
            
            if mat_choice == 0:
                idx = np.random.randint(0, n*m)
                r = np.random.randint(0, R)
                old_val = U[idx, r]
                # Try changing to a different value
                candidates = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(candidates)
                U[idx, r] = new_val
                new_err, _ = compute_error_fast(U, V, W, R)
                if new_err < current_err:
                    current_err = new_err
                    if current_err == 0:
                        return U, V, W
                else:
                    U[idx, r] = old_val
                    
            elif mat_choice == 1:
                idx = np.random.randint(0, m*p)
                r = np.random.randint(0, R)
                old_val = V[idx, r]
                candidates = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(candidates)
                V[idx, r] = new_val
                new_err, _ = compute_error_fast(U, V, W, R)
                if new_err < current_err:
                    current_err = new_err
                    if current_err == 0:
                        return U, V, W
                else:
                    V[idx, r] = old_val
                    
            else:
                idx = np.random.randint(0, p*n)
                r = np.random.randint(0, R)
                old_val = W[idx, r]
                candidates = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(candidates)
                W[idx, r] = new_val
                new_err, _ = compute_error_fast(U, V, W, R)
                if new_err < current_err:
                    current_err = new_err
                    if current_err == 0:
                        return U, V, W
                else:
                    W[idx, r] = old_val
            
            # Every 50 iterations, try a few coordinated moves
            if it % 50 == 49:
                for _ in range(3):
                    mat1 = np.random.choice(3)
                    mat2 = np.random.choice(3)
                    if mat1 == 0:
                        idx1 = np.random.randint(0, n*m)
                        r1 = np.random.randint(0, R)
                        old1 = U[idx1, r1]
                    elif mat1 == 1:
                        idx1 = np.random.randint(0, m*p)
                        r1 = np.random.randint(0, R)
                        old1 = V[idx1, r1]
                    else:
                        idx1 = np.random.randint(0, p*n)
                        r1 = np.random.randint(0, R)
                        old1 = W[idx1, r1]
                    
                    if mat2 == 0:
                        idx2 = np.random.randint(0, n*m)
                        r2 = np.random.randint(0, R)
                        old2 = U[idx2, r2]
                    elif mat2 == 1:
                        idx2 = np.random.randint(0, m*p)
                        r2 = np.random.randint(0, R)
                        old2 = V[idx2, r2]
                    else:
                        idx2 = np.random.randint(0, p*n)
                        r2 = np.random.randint(0, R)
                        old2 = W[idx2, r2]
                    
                    # Make two changes
                    if mat1 == 0:
                        U[idx1, r1] = random.choice([-1, 0, 1])
                    elif mat1 == 1:
                        V[idx1, r1] = random.choice([-1, 0, 1])
                    else:
                        W[idx1, r1] = random.choice([-1, 0, 1])
                    
                    if mat2 == 0:
                        U[idx2, r2] = random.choice([-1, 0, 1])
                    elif mat2 == 1:
                        V[idx2, r2] = random.choice([-1, 0, 1])
                    else:
                        W[idx2, r2] = random.choice([-1, 0, 1])
                    
                    new_err, _ = compute_error_fast(U, V, W, R)
                    if new_err < current_err:
                        current_err = new_err
                        if current_err == 0:
                            return U, V, W
                    else:
                        # Revert
                        if mat1 == 0:
                            U[idx1, r1] = old1
                        elif mat1 == 1:
                            V[idx1, r1] = old1
                        else:
                            W[idx1, r1] = old1
                        
                        if mat2 == 0:
                            U[idx2, r2] = old2
                        elif mat2 == 1:
                            V[idx2, r2] = old2
                        else:
                            W[idx2, r2] = old2
        
        return None

    # Try ranks from 59 down to 45
    init_types = ['random', 'sparse', 'structured']
    
    for R in range(59, 44, -1):
        if time.time() > deadline - 10:
            break
        
        for init_type in init_types:
            if time.time() > deadline - 10:
                break
            
            for restart in range(3):
                if time.time() > deadline - 10:
                    break
                
                result = try_rank(R, max_iters=1500, init_type=init_type)
                if result is not None:
                    U, V, W = result
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r in range(R):
                                    s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                                if s != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better solution found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
