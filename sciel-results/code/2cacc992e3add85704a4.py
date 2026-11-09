import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple
    # T is a 3-tensor of shape (n, m, p)
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Initialize with schoolbook decomposition (rank = n*m*p = 50)
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
    
    # Try to reduce rank using randomized optimization
    # Start with various initial ranks and try to find exact decompositions
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try different target ranks
    target_ranks = [40, 45, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize factors with random values
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent to minimize reconstruction error
            lr = 0.01
            tol = 1e-10
            max_iter = 500
            
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Compute reconstruction: for each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # This is expensive, so we'll use a smarter approach
                
                # Compute the reconstructed tensor
                recon = np.zeros((n, m, p))
                for r in range(target_rank):
                    # U[:, r] is vector of length n*m, reshape to (n, m)
                    # V[:, r] is vector of length m*p, reshape to (m, p)
                    # W[:, r] is vector of length p*n, reshape to (p, n)
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    
                    # For each (i,j,k): Ur[i,j] * Vr[j,k] * Wr[k,i]
                    # This is a rank-1 tensor
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                recon[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
                
                # Target tensor
                target = np.ones((n, m, p))
                error = np.sum((recon - target) ** 2)
                
                if error < tol:
                    break
                
                # Compute gradients (simplified - use finite differences for correctness)
                # For efficiency, we'll use a simpler update rule
                
                # Update U
                for r in range(target_rank):
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    grad_U = np.zeros((n, m))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                grad_U[i, j] += 2 * (recon[i, j, k] - 1) * Vr[j, k] * Wr[k, i]
                    U_r[:, r] -= lr * grad_U.flatten()
                
                # Update V
                for r in range(target_rank):
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    grad_V = np.zeros((m, p))
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                grad_V[j, k] += 2 * (recon[i, j, k] - 1) * Ur[i, j] * Wr[k, i]
                    V_r[:, r] -= lr * grad_V.flatten()
                
                # Update W
                for r in range(target_rank):
                    Ur = U_r[:, r].reshape(n, m)
                    Vr = V_r[:, r].reshape(m, p)
                    Wr = W_r[:, r].reshape(p, n)
                    grad_W = np.zeros((p, n))
                    for k in range(p):
                        for i in range(n):
                            for j in range(m):
                                grad_W[k, i] += 2 * (recon[i, j, k] - 1) * Ur[i, j] * Vr[j, k]
                    W_r[:, r] -= lr * grad_W.flatten()
            
            # Check final error
            recon = np.zeros((n, m, p))
            for r in range(target_rank):
                Ur = U_r[:, r].reshape(n, m)
                Vr = V_r[:, r].reshape(m, p)
                Wr = W_r[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
            
            target = np.ones((n, m, p))
            final_error = np.sum((recon - target) ** 2)
            
            if final_error < tol and target_rank < best_rank:
                best_rank = target_rank
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        if best_rank <= target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
