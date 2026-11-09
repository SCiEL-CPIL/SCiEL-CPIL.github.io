"""Combined program: gradient-based tensor decomposition search with fast verification."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin from 120s limit
    
    # Schoolbook decomposition as fallback
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, sb_rank = _schoolbook()
    
    # Fast residual computation using vectorized ops
    def _residual(U, V, W, target_rank):
        recon = np.zeros((n, m, p))
        for c in range(target_rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
        return np.linalg.norm(recon - np.ones((n, m, p)))
    
    # Gradient computation (vectorized)
    def _gradients(U, V, W, diff, target_rank):
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for c in range(target_rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            
            # U_grad[i,j,c] = 2 * sum_k diff[i,j,k] * V_c[j,k] * W_c[k,i]
            # Vectorized: for each (i,j), sum over k
            # diff[i,j,:] * (V_c[j,:] * W_c[:,i])
            # Shape: (n, m, p) * (m, p) broadcast... need careful indexing
            # U_grad_c[i,j] = sum_k diff[i,j,k] * V_c[j,k] * W_c[k,i]
            # = einsum('ijk,jk,ki->ij', diff, V_c, W_c)
            U_grad_c = np.einsum('ijk,jk,ki->ij', diff, V_c, W_c)
            U_grad[:, c] = 2.0 * U_grad_c.flatten()
            
            # V_grad[j,k,c] = 2 * sum_i diff[i,j,k] * U_c[i,j] * W_c[k,i]
            V_grad_c = np.einsum('ijk,ij,ki->jk', diff, U_c, W_c)
            V_grad[:, c] = 2.0 * V_grad_c.flatten()
            
            # W_grad[k,i,c] = 2 * sum_j diff[i,j,k] * U_c[i,j] * V_c[j,k]
            W_grad_c = np.einsum('ijk,ij,jk->ki', diff, U_c, V_c)
            W_grad[:, c] = 2.0 * W_grad_c.flatten()
        
        return U_grad, V_grad, W_grad
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = sb_rank
    best_residual = 0.0
    
    np.random.seed(42)
    
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)  # max(9, 15, 15) = 15
    
    # Try ranks from lower_bound+1 up to sb_rank-1
    for target_rank in range(lower_bound + 1, sb_rank):
        if time.time() > deadline - 20:
            break
        
        # Multiple restarts
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            scale = 0.5
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            best_local_residual = np.inf
            stagnant = 0
            
            for iteration in range(300):
                if time.time() > deadline - 15:
                    break
                
                # Compute reconstruction
                recon = np.zeros((n, m, p))
                for c in range(target_rank):
                    U_c = U[:, c].reshape(n, m)
                    V_c = V[:, c].reshape(m, p)
                    W_c = W[:, c].reshape(p, n)
                    recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
                
                diff = recon - np.ones((n, m, p))
                residual = np.linalg.norm(diff)
                
                if residual < 1e-10:
                    # Found exact solution!
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    best_residual = residual
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
                if residual < best_local_residual:
                    best_local_residual = residual
                    stagnant = 0
                else:
                    stagnant += 1
                
                if stagnant > 50:
                    break
                
                # Compute gradients
                U_grad, V_grad, W_grad = _gradients(U, V, W, diff, target_rank)
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Decay learning rate
                lr *= 0.995
                if lr < 0.001:
                    lr = 0.001
            
            # Check if this restart found something good
            if best_local_residual < 1e-6 and target_rank < best_rank:
                # Do a polish pass with tighter tolerance
                lr = 0.005
                for polish_iter in range(200):
                    if time.time() > deadline - 10:
                        break
                    
                    recon = np.zeros((n, m, p))
                    for c in range(target_rank):
                        U_c = U[:, c].reshape(n, m)
                        V_c = V[:, c].reshape(m, p)
                        W_c = W[:, c].reshape(p, n)
                        recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
                    
                    diff = recon - np.ones((n, m, p))
                    residual = np.linalg.norm(diff)
                    
                    if residual < 1e-12:
                        break
                    
                    U_grad, V_grad, W_grad = _gradients(U, V, W, diff, target_rank)
                    U -= lr * U_grad
                    V -= lr * V_grad
                    W -= lr * W_grad
                    lr *= 0.999
                
                final_residual = _residual(U, V, W, target_rank)
                if final_residual < 1e-8 and target_rank < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    best_residual = final_residual
                    break
    
    # Convert to lists and return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
