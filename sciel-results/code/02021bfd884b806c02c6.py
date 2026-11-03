import numpy as np
import time
import sys

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute index mappings
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    def compute_residual(U, V, W):
        """Compute sum of squared errors over all (i,j,k)."""
        R = U.shape[1]
        residual = 0.0
        for u_idx, v_idx, w_idx in indices:
            # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
            residual += (val - 1.0) ** 2
        return residual
    
    # Schoolbook solution
    seed_U = np.zeros((size_u, n*m*p))
    seed_V = np.zeros((size_v, n*m*p))
    seed_W = np.zeros((size_w, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                seed_U[i*m+j, col] = 1
                seed_V[j*p+k, col] = 1
                seed_W[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 130  # Leave 20s safety margin
    
    best_U, best_V, best_W = seed_U.copy(), seed_V.copy(), seed_W.copy()
    best_rank = n*m*p
    
    # Try to reduce rank
    for R in range(n*m*p - 1, 40, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random restarts
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with first R columns + noise
            U = seed_U[:, :R].copy()
            V = seed_V[:, :R].copy()
            W = seed_W[:, :R].copy()
            
            # Add small random noise
            noise = 0.05
            U += np.random.normal(0, noise, U.shape)
            V += np.random.normal(0, noise, V.shape)
            W += np.random.normal(0, noise, W.shape)
            
            # Gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute residual
                res = compute_residual(U, V, W)
                if res < 1e-10:
                    break
                
                # Compute gradients numerically (expensive but simple)
                # Or use analytical gradients
                
                # Analytical gradient:
                # d/dU[u,r] of sum_{i,j,k} (sum_r' U[i*m+j,r'] V[j*p+k,r'] W[k*n+i,r'] - 1)^2
                # = 2 * sum_{i,j,k: i*m+j=u} (val_{i,j,k} - 1) * V[j*p+k,r] * W[k*n+i,r]
                
                # This is complex. Let's use a simpler approach:
                # For each column r, compute the contribution to the gradient.
                
                # Actually, let's just use a simple heuristic:
                # If residual is large, increase learning rate; if oscillating, decrease.
                
                # Compute gradient for U
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for u_idx, v_idx, w_idx in indices:
                    # val = sum_r U[u_idx,r] V[v_idx,r] W[w_idx,r]
                    uv = U[u_idx] * V[v_idx]
                    val = np.dot(uv, W[w_idx])
                    err = val - 1.0
                    
                    # d/dU[u_idx, r] = 2 * err * V[v_idx, r] * W[w_idx, r]
                    grad_U[u_idx] += 2 * err * V[v_idx] * W[w_idx]
                    grad_V[v_idx] += 2 * err * U[u_idx] * W[w_idx]
                    grad_W[w_idx] += 2 * err * U[u_idx] * V[v_idx]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if step > 0 and res < prev_res * 0.9:
                    lr *= 1.1
                elif step > 0 and res > prev_res * 1.1:
                    lr *= 0.9
                
                prev_res = res
            
            # Check if we found a solution
            final_res = compute_residual(U, V, W)
            if final_res < 1e-8:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = R
                break  # Found a solution at this rank, try lower
        
        if time.time() - start_time > time_limit:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
