import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    def validate(U, V, W, tol=1e-9):
        """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j, :]
                    v_val = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the standard rank-45 schoolbook decomposition"""
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        return U, V, W
    
    def random_init(rank, seed=None):
        """Initialize random factor matrices"""
        rng = np.random.RandomState(seed)
        U = rng.randn(N, rank) * 0.1
        V = rng.randn(M, rank) * 0.1
        W = rng.randn(P, rank) * 0.1
        return U, V, W
    
    def gradient_descent(U, V, W, rank, max_iter=5000, lr=0.01, seed=None):
        """Optimize factors to minimize error in tensor decomposition"""
        rng = np.random.RandomState(seed)
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute index arrays
        idx_u = np.array([i*m + j for i in range(n) for j in range(m) for k in range(p)])
        idx_v = np.array([j*p + k for i in range(n) for j in range(m) for k in range(p)])
        idx_w = np.array([k*n + i for i in range(n) for j in range(m) for k in range(p)])
        
        for it in range(max_iter):
            # Compute gradients
            # Loss = sum_{i,j,k} (1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            # dLoss/dU[i*m+j, r] = -2 * sum_{(i',j',k') where i'*m+j'=i*m+j} (1 - s_{i'j'k'}) * V[j'*p+k',r] * W[k'*n+i',r]
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        
                        # s = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        err = 1.0 - s
                        
                        # Gradient for this term
                        # d/dU[u_idx, r] = -2 * err * V[v_idx, r] * W[w_idx, r]
                        grad_U[u_idx, :] += -2.0 * err * V[v_idx, :] * W[w_idx, :]
                        grad_V[v_idx, :] += -2.0 * err * U[u_idx, :] * W[w_idx, :]
                        grad_W[w_idx, :] += -2.0 * err * U[u_idx, :] * V[v_idx, :]
            
            # Update with adaptive learning rate
            loss = np.sum((1.0 - np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :], axis=1))**2)
            if it % 100 == 0 and it > 0:
                pass  # Could log here
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if loss < 1e-10:
                break
            
            # Check time
            if time.time() > deadline - 20:
                break
        
        return U, V, W, loss
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Try ranks from 15 up to 44
    
    for rank in range(15, min(44, n*m*p)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            seed = hash((rank, attempt)) % (2**32)
            U_init, V_init, W_init = random_init(rank, seed=seed)
            
            # Run gradient descent
            U_opt, V_opt, W_opt, loss = gradient_descent(
                U_init, V_init, W_opt if False else W_init, 
                rank, max_iter=3000, lr=0.005, seed=seed
            )
            
            # Validate
            if validate(U_opt, V_opt, W_opt, tol=1e-6):
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = rank
                break  # Found valid decomposition at this rank
        
        # If we found a valid decomposition at this rank, we can stop
        # (since we're going from low to high)
        if best_rank == rank:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
