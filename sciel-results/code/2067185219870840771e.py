import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    target_rank = 12
    
    # Shapes
    rows_U = n * m  # 6
    rows_V = m * p  # 12
    rows_W = p * n  # 8
    
    # Total variables
    total_vars = rows_U * target_rank + rows_V * target_rank + rows_W * target_rank
    
    # Grid of allowed values
    grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    grid_size = len(grid)
    
    # Pre-compute index mappings for fast access
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # We represent the state as a flat array of indices into the grid
    # or directly the values. Values are easier for calculation.
    
    def compute_error(U, V, W):
        # U: (6, 12), V: (12, 12), W: (8, 12)
        # T_comp[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    diff = s - 1.0
                    err += diff * diff
        return err

    def try_local_search(start_state, max_iters=5000):
        # start_state: flat array of floats
        U = start_state[:rows_U*target_rank].reshape(rows_U, target_rank).copy()
        V = start_state[rows_U*target_rank:rows_U*target_rank+rows_V*target_rank].reshape(rows_V, target_rank).copy()
        W = start_state[rows_U*target_rank+rows_V*target_rank:].reshape(rows_W, target_rank).copy()
        
        current_err = compute_error(U, V, W)
        
        for _ in range(max_iters):
            # Pick a random variable
            var_idx = np.random.randint(0, total_vars)
            
            # Determine which matrix and position
            if var_idx < rows_U * target_rank:
                idx = var_idx
                mat = U
                r, c = divmod(idx, target_rank)
            elif var_idx < rows_U * target_rank + rows_V * target_rank:
                idx = var_idx - rows_U * target_rank
                mat = V
                r, c = divmod(idx, target_rank)
            else:
                idx = var_idx - rows_U * target_rank - rows_V * target_rank
                mat = W
                r, c = divmod(idx, target_rank)
                
            old_val = mat[r, c]
            
            # Try changing to a random other grid value
            new_val = np.random.choice(grid, replace=False)
            if new_val == old_val:
                continue
                
            mat[r, c] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-9:
                    return U, V, W, current_err
            else:
                mat[r, c] = old_val
                
        return U, V, W, current_err

    best_result = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Random initialization from grid
        init_state = np.random.choice(grid, size=total_vars)
        
        U, V, W, err = try_local_search(init_state, max_iters=1000)
        
        if err < best_err:
            best_err = err
            best_result = (U, V, W)
            
        if best_err < 1e-9:
            break
            
        restarts += 1

    if best_result is not None and best_err < 1e-9:
        U, V, W = best_result
        # Ensure exactness by rounding to grid (should already be there)
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Final verification
        if compute_error(U, V, W) < 1e-9:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
