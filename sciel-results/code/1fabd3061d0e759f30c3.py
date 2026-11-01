import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # We want to find a rank lower than schoolbook_rank
    # Try ranks from 15 up to schoolbook_rank
    # Using a deterministic local search with fixed seeds
    
    target_val = 2  # We scale by 2 to allow half-integers
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
                
    def calc_error(U, V, W):
        err = 0
        for u_idx, v_idx, w_idx in indices:
            s = 0
            for r in range(len(U[0])):
                s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            diff = s - target_val
            err += diff * diff
        return err

    def try_solve(rank, seed):
        np.random.seed(seed)
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        # Initialize with random values
        for _ in range(rank * 4):
            U[np.random.randint(0, n*m), np.random.randint(0, rank)] = np.random.choice([-1, 1])
            V[np.random.randint(0, m*p), np.random.randint(0, rank)] = np.random.choice([-1, 1])
            W[np.random.randint(0, p*n), np.random.randint(0, rank)] = np.random.choice([-1, 1])
            
        current_err = calc_error(U, V, W)
        
        # Local search
        max_iters = 50000
        for _ in range(max_iters):
            if current_err == 0:
                return U, V, W
            
            # Select a matrix to update
            mat_type = np.random.randint(3)
            if mat_type == 0:
                mat = U
                dim = n*m
            elif mat_type == 1:
                mat = V
                dim = m*p
            else:
                mat = W
                dim = p*n
                
            row = np.random.randint(0, dim)
            col = np.random.randint(0, rank)
            
            old_val = mat[row, col]
            best_val = old_val
            best_err = current_err
            
            # Try all possible integer values in a small range
            for new_val in range(-3, 4):
                if new_val == old_val:
                    continue
                mat[row, col] = new_val
                new_err = calc_error(U, V, W)
                if new_err < best_err:
                    best_err = new_err
                    best_val = new_val
                mat[row, col] = old_val
            
            if best_val != old_val:
                mat[row, col] = best_val
                current_err = best_err
            else:
                # Random kick if stuck
                if np.random.rand() < 0.05:
                    mat[row, col] = np.random.choice([-3, -2, -1, 0, 1, 2, 3])
                    current_err = calc_error(U, V, W)
                    
        return None

    # Try to find a solution for ranks 15, 16, 17
    for rank in [15, 16, 17]:
        for seed in range(20):
            res = try_solve(rank, seed)
            if res:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
