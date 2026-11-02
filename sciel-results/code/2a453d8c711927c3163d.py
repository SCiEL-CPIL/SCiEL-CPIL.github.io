"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor T[i,j,k] = 1 if C[i,k] involves A[i,j]*B[j,k]
    # T is a 3D array of shape (n*m, m*p, p*n) in the flattened convention
    # Actually, the tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook rank 128
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    # Vectorized reconstruction
    def reconstruct(U_t, V_t, W_t, tensor_shape):
        """Reconstruct tensor from factor matrices using vectorized ops."""
        d1, d2, d3 = tensor_shape
        r = U_t.shape[1]
        recon = np.zeros((d1, d2, d3), dtype=float)
        for c in range(r):
            recon += np.einsum('i,j,k->ijk', U_t[:, c], V_t[:, c], W_t[:, c])
        return recon
    
    # Vectorized gradient computation
    def compute_gradients(tensor, U_t, V_t, W_t, recon):
        """Compute gradients for U, V, W."""
        d1, d2, d3 = tensor.shape
        r = U_t.shape[1]
        diff = tensor - recon
        
        grad_U = np.zeros_like(U_t)
        grad_V = np.zeros_like(V_t)
        grad_W = np.zeros_like(W_t)
        
        for c in range(r):
            # grad_U[:,c] = sum_{b,k} diff[:,b,k] * V_t[b,c] * W_t[k,c]
            grad_U[:, c] = np.einsum('ijk,j,k->i', diff, V_t[:, c], W_t[:, c])
            # grad_V[:,c] = sum_{a,k} diff[a,:,k] * U_t[a,c] * W_t[k,c]
            grad_V[:, c] = np.einsum('ijk,i,k->j', diff, U_t[:, c], W_t[:, c])
            # grad_W[:,c] = sum_{a,b} diff[a,b,:] * U_t[a,c] * V_t[b,c]
            grad_W[:, c] = np.einsum('ijk,i,j->k', diff, U_t[:, c], V_t[:, c])
        
        return grad_U, grad_V, grad_W
    
    def try_decompose(tensor, dim1, dim2, dim3, target_rank, max_iter=200, seed=42, deadline=None):
        """Try to find a decomposition of the tensor with given target rank using vectorized gradient descent."""
        np.random.seed(seed)
        U_t = np.random.randn(dim1, target_rank) * 0.5
        V_t = np.random.randn(dim2, target_rank) * 0.5
        W_t = np.random.randn(dim3, target_rank) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        recon = reconstruct(U_t, V_t, W_t, (dim1, dim2, dim3))
        loss = np.sum((tensor - recon)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        lr = 0.05
        patience = 0
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(tensor, U_t, V_t, W_t, recon)
            
            U_t -= lr * grad_U
            V_t -= lr * grad_V
            W_t -= lr * grad_W
            
            recon = reconstruct(U_t, V_t, W_t, (dim1, dim2, dim3))
            loss = np.sum((tensor - recon)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                patience = 0
                if best_loss < 1e-8:
                    break
            else:
                patience += 1
                # Adaptive learning rate
                if patience > 20:
                    lr *= 0.95
                    patience = 0
            
            if it % 50 == 0 and it > 0:
                # Try to reset with smaller lr if stuck
                if best_loss > 1.0:
                    lr = max(lr * 0.8, 0.001)
        
        return best_U, best_V, best_W, best_loss
    
    # Try to find lower rank decompositions
    # First try rank 96, 92, 88, etc.
    for target_rank in [96, 92, 88, 84, 80]:
        if time.time() > deadline - 10:
            break
        
        best_loss = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            U_r, V_r, W_r, loss = try_decompose(
                T, n*m, m*p, p*n, target_rank, 
                max_iter=100, seed=seed, deadline=deadline - 5
            )
            
            if loss < best_loss:
                best_loss = loss
                best_U_r, best_V_r, best_W_r = U_r, V_r, W_r
            
            if best_loss < 1e-8:
                break
        
        if best_loss < 1e-8:
            best_rank = target_rank
            best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
            break
    
    # Return the best result found
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
