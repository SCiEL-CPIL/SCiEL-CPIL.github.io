import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Indices
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
    
    # Initial Rank
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        U[idx_u[r], r] = 1
        V[idx_v[r], r] = 1
        W[idx_w[r], r] = 1
        
    # Target Tensor (sparse representation)
    # T[a, b, c] = 1 if (a,b,c) is in the list
    # We only need to check the non-zero entries for error.
    # Error = sum over (a,b,c) in list of (1 - sum_r U[a,r]V[b,r]W[c,r])^2
    # Since the tensor is 0 elsewhere, and our decomposition will likely be 0 elsewhere if we start from schoolbook?
    # Actually, ALS might introduce non-zeros elsewhere.
    # But the target is 0 elsewhere.
    # So we must also penalize non-zero entries at (a,b,c) NOT in the list.
    # This is hard to do efficiently if we don't store the full tensor.
    # However, the full tensor is 15x30x18 = 8100 entries. This is small.
    # Let's create the full target tensor T.
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T[idx_u[r], idx_v[r], idx_w[r]] = 1.0
        
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    time_limit = 120 # seconds
    
    while best_rank > 1 and (time.time() - start_time) < time_limit:
        current_rank = best_rank
        # Try to remove one term
        success = False
        # Random order of columns to try
        cols = np.arange(current_rank)
        np.random.shuffle(cols)
        
        for c in cols:
            if (time.time() - start_time) > time_limit:
                break
                
            # Copy current best
            U_curr = best_U[:, :current_rank].copy()
            V_curr = best_V[:, :current_rank].copy()
            W_curr = best_W[:, :current_rank].copy()
            
            # Zero out column c
            U_curr[:, c] = 0
            V_curr[:, c] = 0
            W_curr[:, c] = 0
            
            # Run ALS for a few iterations
            # We want to minimize || T - sum_{r != c} u_r v_r w_r ||^2
            # The term r=c is zero, so we are fitting T with current_rank-1 terms.
            
            for _ in range(20): # ALS iterations
                # Update U
                # For each alpha, solve for U[alpha, r]
                # Actually, we update all r != c simultaneously?
                # No, ALS usually updates one factor at a time, or one vector at a time.
                # Let's update the whole U matrix.
                # Fix V and W.
                # T_alpha_beta_gamma = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                # For each alpha, we have a linear system for U[alpha, :]?
                # No, U[alpha, r] appears in T[alpha, beta, gamma] multiplied by V[beta, r] W[gamma, r].
                # This is a linear least squares for the vector U[alpha, :].
                # Let A_alpha be a matrix where A_alpha[beta * p*n + gamma, r] = V[beta, r] W[gamma, r].
                # Then T[alpha, :, :] (flattened) = A_alpha @ U[alpha, :].
                # We solve U[alpha, :] = (A_alpha^T A_alpha)^{-1} A_alpha^T T[alpha, :, :]
                
                # This is expensive to do for each alpha if we build A_alpha.
                # But dimensions are small.
                
                # Alternative: Update one vector at a time.
                # Let's stick to the "remove one term and re-optimize all others" idea.
                # It's simpler to just run a standard ALS on the remaining terms.
                
                # Let's implement a simple ALS that updates U, then V, then W.
                
                # Update U
                for r in range(current_rank):
                    if r == c: continue
                    # Solve for U[:, r]
                    # T = sum u v w
                    # Fix v_r, w_r.
                    # We want to find u_r to minimize || T - u_r (v_r w_r)^T - Rest ||^2
                    # Let Residual = T - Rest (where Rest is sum of all other terms)
                    # Minimize || Residual - u_r (v_r w_r)^T ||^2
                    # This is not a simple linear solve for u_r because u_r is a vector and (v_r w_r)^T is a matrix.
                    # Actually, it IS a linear solve.
                    # Let M = v_r w_r^T (size m*p x p*n).
                    # We want u_r such that u_r M approx Residual.
                    # Flatten Residual to vector y (size (m*p)*(p*n)).
                    # Flatten M to vector z? No.
                    # Residual[a, b, c] = sum_r' U[a, r'] V[b, r'] W[c, r']
                    # We want to update U[a, r].
                    # For fixed a, b, c:
                    # T[a,b,c] approx sum_{r' != r} U[a, r'] V[b, r'] W[c, r'] + U[a, r] V[b, r] W[c, r]
                    # Let RHS[a, b, c] = T[a,b,c] - sum_{r' != r} U[a, r'] V[b, r'] W[c, r']
                    # We want to find U[a, r] to minimize sum_{b,c} (RHS[a, b, c] - U[a, r] V[b, r] W[c, r])^2
                    # This is a linear regression: y = x * z
                    # y = RHS[a, :, :] (flattened)
                    # z = V[:, r] * W[:, r] (outer product flattened)
                    # x = U[a, r]
                    # x = dot(y, z) / dot(z, z)
                    
                    # This is O(1) per a, if we have RHS and z.
                    # But calculating RHS requires summing over all other r'.
                    # This is O(R) per a.
                    # Total O(R * N * M * P) per update of U[:, r].
                    # Doing this for all r is O(R^2 * N * M * P).
                    # R=90, N=15, M=30, P=18.
                    # 90^2 * 15 * 30 * 18 ~ 8100 * 8100 ~ 65 million ops.
                    # This is fast in NumPy.
                    
                # Let's implement this specific ALS step.
                
                # Precompute "Rest" for each r?
                # Rest_r = T - sum_{r' != r} u_{r'} v_{r'} w_{r'}
                # This is hard to compute efficiently for all r.
                
                # Let's just do standard ALS on the full matrix, but with column c zeroed.
                # We can use `scipy.linalg` or manual numpy.
                
                # Given the small size, let's just use a generic ALS implementation.
                
            # After ALS, check error
            # Reconstruct T_approx
            T_approx = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
            error = np.sum((T - T_approx)**2)
            
            if error < 1e-8:
                best_rank = current_rank - 1
                best_U = U_curr[:, :best_rank].copy()
                best_V = V_curr[:, :best_rank].copy()
                best_W = W_curr[:, :best_rank].copy()
                success = True
                break # Restart the while loop with new best_rank
                
        if not success:
            break # No more reductions found
            
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
