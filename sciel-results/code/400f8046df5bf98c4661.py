import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145

    # Build the rank-55 Strassen-based decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Validate the base decomposition
    def validate(U_v, V_v, W_v, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_v.shape[1]):
                        s += U_v[i*m+j, r] * V_v[j*p+k, r] * W_v[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not validate(U_np, V_np, W_np):
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    
    # Try to reduce rank by removing columns
    best_rank = rank
    
    def try_remove_column(U_c, V_c, W_c, r):
        """Check if removing column r still gives valid decomposition."""
        U_t = np.delete(U_c, r, axis=1)
        V_t = np.delete(V_c, r, axis=1)
        W_t = np.delete(W_c, r, axis=1)
        return validate(U_t, V_t, W_t)
    
    # Attempt removals
    for attempt in range(2):
        if time.time() > deadline - 20:
            break
        for r in range(U_np.shape[1]):
            if time.time() > deadline - 10:
                break
            if try_remove_column(U_np, V_np, W_np, r):
                U_np = np.delete(U_np, r, axis=1)
                V_np = np.delete(V_np, r, axis=1)
                W_np = np.delete(W_np, r, axis=1)
                best_rank = U_np.shape[1]
                break
    
    # Try ALS-based optimization to find lower rank decomposition
    def als_optimize(U_init, V_init, W_init, target_rank, max_iters=50, tol=1e-10):
        """Attempt to find a valid rank-target_rank decomposition using ALS."""
        if time.time() > deadline - 30:
            return None
        
        # Initialize with random perturbation of the existing factors
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        # Add small random perturbation to escape local minima
        np.random.seed(42)
        U += 0.01 * np.random.randn(*U.shape)
        V += 0.01 * np.random.randn(*V.shape)
        W += 0.01 * np.random.randn(*W.shape)
        
        for it in range(max_iters):
            if time.time() > deadline - 25:
                return None
            
            # Update U
            for r in range(target_rank):
                # For each r, solve for U[:,r] such that
                # sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
                # Fix V and W, solve for U
                A_mat = np.zeros((n*m, n*m))
                b_vec = np.zeros(n*m)
                
                for i in range(n):
                    for j in range(m):
                        row_idx = i*m + j
                        for i2 in range(n):
                            for j2 in range(m):
                                col_idx = i2*m + j2
                                # Coefficient of U[row_idx, r] in the constraint for (i2,j2,k)
                                # We need: sum_k V[j2*p+k, r] * W[k*n+i2, r] * U[i2*m+j2, r']
                                # This is complex, so use a different approach
                                pass
                
                # Alternative: use gradient-based approach
                # Compute residual
                residual = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r2 in range(target_rank):
                                s += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                            residual[i,j,k] = s - 1.0
                
                if np.max(np.abs(residual)) < tol:
                    break
                
                # Gradient descent on U
                lr = 0.001
                for i in range(n):
                    for j in range(m):
                        grad = 0.0
                        for k in range(p):
                            grad += residual[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        U[i*m+j, r] -= lr * grad
                
                # Gradient descent on V
                for j in range(m):
                    for k in range(p):
                        grad = 0.0
                        for i in range(n):
                            grad += residual[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                        V[j*p+k, r] -= lr * grad
                
                # Gradient descent on W
                for k in range(p):
                    for i in range(n):
                        grad = 0.0
                        for j in range(m):
                            grad += residual[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                        W[k*n+i, r] -= lr * grad
        
        # Round to half-integers and validate
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if validate(U_round, V_round, W_round, tol=1e-9):
            return U_round, V_round, W_round
        
        return None
    
    # Try target ranks from current down to 23
    for target_rank in range(best_rank - 1, 22, -1):
        if time.time() > deadline - 40:
            break
        
        result = als_optimize(U_np, V_np, W_np, target_rank)
        if result is not None:
            U_np, V_np, W_np = result
            best_rank = target_rank
            break
    
    # Final validation
    if not validate(U_np, V_np, W_np):
        # Fallback to the best valid decomposition found
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
    
    # Convert to lists
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    # Ensure entries are integers or half-integers
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}
