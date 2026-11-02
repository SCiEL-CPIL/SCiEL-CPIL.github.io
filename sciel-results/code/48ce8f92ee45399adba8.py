import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds to be safe within 150s limit

    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 72
    
    # Precompute indices for the 72 equations
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m + j, j*p + k, k*n + i))
    indices = np.array(indices)
    
    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute sum_r U[i,j] V[j,k] W[k,i]
        # Using einsum: 'ir, jr, kr -> r' is not direct.
        # We want for each (i,j,k) in indices: sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        
        U_idx = U[indices[:, 0], :] # (72, R)
        V_idx = V[indices[:, 1], :] # (72, R)
        W_idx = W[indices[:, 2], :] # (72, R)
        
        # Sum over r
        prod = U_idx * V_idx * W_idx # (72, R)
        sums = np.sum(prod, axis=1) # (72,)
        
        errors = sums - 1.0
        return np.sum(errors**2)

    def compute_gradients(U, V, W):
        # Returns gradients for U, V, W
        # E = sum_eq (sum_r u v w - 1)^2
        # dE/du_{a,r} = 2 * sum_eq (sum - 1) * v_{eq, r} * w_{eq, r} * delta(u_idx == a)
        
        U_idx = U[indices[:, 0], :]
        V_idx = V[indices[:, 1], :]
        W_idx = W[indices[:, 2], :]
        
        sums = np.sum(U_idx * V_idx * W_idx, axis=1)
        err = sums - 1.0 # (72,)
        
        # Gradients
        # dU: (n*m, R)
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # For each rank r, and each equation eq:
        # term = 2 * err[eq] * V_idx[eq, r] * W_idx[eq, r]
        # Add to dU[U_idx[eq], r]
        
        # Vectorized approach:
        # We can scatter add.
        # term_u[eq, r] = 2 * err[eq] * V_idx[eq, r] * W_idx[eq, r]
        term_u = 2.0 * err[:, np.newaxis] * V_idx * W_idx
        term_v = 2.0 * err[:, np.newaxis] * U_idx * W_idx
        term_w = 2.0 * err[:, np.newaxis] * U_idx * V_idx
        
        # Scatter add
        for r in range(U.shape[1]):
            np.add.at(dU, indices[:, 0], term_u[:, r])
            np.add.at(dV, indices[:, 1], term_v[:, r])
            np.add.at(dW, indices[:, 2], term_w[:, r])
            
        return dU, dV, dW

    def optimize(rank, seed):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Momentum
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        mu = 0.9
        lr = 0.1
        
        min_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(500):
            if time.time() > deadline - 5:
                break
            
            err = compute_error(U, V, W)
            if err < min_err:
                min_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            dU, dV, dW = compute_gradients(U, V, W)
            
            vU = mu * vU - lr * dU
            vV = mu * vV - lr * dV
            vW = mu * vW - lr * dW
            
            U += vU
            V += vV
            W += vW
            
            # Optional: normalize to prevent divergence?
            # For now, rely on learning rate.
            
        # Try to snap to half-integers
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        
        err_r = compute_error(U_r, V_r, W_r)
        
        if err_r < 1e-5:
            return U_r, V_r, W_r, err_r
        else:
            return best_U, best_V, best_W, min_err

    # Try ranks from 1 upwards? No, start from a reasonable guess.
    # Theoretical lower bound is roughly (n*p + m*p + n*m) / 2? 
    # For 3x3x3, rank is 23 (Laderman) or 21 (Strassen).
    # For 3x3x8, it's larger.
    # Schoolbook is 72.
    # Let's try ranks 10 to 72.
    
    # We will run multiple restarts for each rank, prioritizing lower ranks.
    # But finding a low rank is hard.
    # We will iterate ranks from 1 to 72.
    # For each rank, we do a few restarts.
    
    # To save time, we can do a coarse search first.
    # Let's try a set of candidate ranks.
    candidate_ranks = list(range(10, 73))
    
    # We might find that even rank 30 is hard.
    # We'll spend equal time per rank? No, lower rank is more valuable.
    # But if we can't find rank 10, we need to find rank 20.
    
    # Strategy:
    # Loop through ranks.
    # For each rank, do 5 restarts.
    # If found, stop.
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            U_c, V_c, W_c, err_c = optimize(rank, seed)
            
            if err_c < 1e-5:
                # Verify exactness with higher precision logic?
                # The problem allows half-integers.
                # If err is small, it's likely good.
                best_rank = rank
                best_U, best_V, best_W = U_c, V_c, W_c
                found = True
                break
        
        if found:
            break
            
    # If no low rank found, fall back to schoolbook
    if best_rank == n * m * p:
        # Construct schoolbook
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
        best_U, best_V, best_W = U, V, W

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
