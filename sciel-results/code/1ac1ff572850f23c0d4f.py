import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Try ranks from low to high
    # Prioritize 19, 20, 21, 22 as they are more likely to be found
    for target_rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline)
        if result is not None:
            return result
    
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


def _try_rank(target_rank, n, m, p, deadline):
    """
    Try to find a rank-target_rank decomposition using structured search.
    """
    rows_U = n * m  # 6
    rows_V = m * p  # 12
    rows_W = p * n  # 8
    
    # Expanded grid of allowed values
    grid = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    def compute_error(U, V, W):
        """Compute sum of squared errors over all (i,j,k) triples."""
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
    
    def try_local_search(U, V, W, max_iters=5000):
        """Local search to improve the decomposition."""
        current_err = compute_error(U, V, W)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random variable
            var_idx = np.random.randint(0, rows_U * target_rank + rows_V * target_rank + rows_W * target_rank)
            
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
            new_val = np.random.choice(grid)
            if new_val == old_val:
                continue
                
            mat[r, c] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-10:
                    return U, V, W, current_err
            else:
                mat[r, c] = old_val
                
        return U, V, W, current_err
    
    best_result = None
    best_err = float('inf')
    
    # Multiple restarts with different initialization strategies
    n_restarts = 100
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization from grid
        U = np.random.choice(grid, size=(rows_U, target_rank))
        V = np.random.choice(grid, size=(rows_V, target_rank))
        W = np.random.choice(grid, size=(rows_W, target_rank))
        
        U, V, W, err = try_local_search(U, V, W, max_iters=2000)
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
            
        if best_err < 1e-10:
            break
    
    if best_result is not None and best_err < 1e-10:
        U, V, W = best_result
        # Round to nearest grid value
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Final verification
        if compute_error(U, V, W) < 1e-10:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None
