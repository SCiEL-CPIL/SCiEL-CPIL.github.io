import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > 1e-7:
                        return False
        return True

    def objective_and_grad(U, V, W):
        """Compute objective value and gradients for the tensor decomposition problem."""
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        obj = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err = val - 1
                    obj += err * err
                    
                    grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
                    grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
                    grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
        
        return obj, grad_U, grad_V, grad_W

    def gradient_descent(U, V, W, max_steps=2000, lr_init=0.01, lr_decay=0.999, tol=1e-10):
        """Run gradient descent with learning rate decay."""
        lr = lr_init
        best_obj = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            obj, grad_U, grad_V, grad_W = objective_and_grad(U, V, W)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if obj < tol:
                break
            
            # Adaptive learning rate
            grad_norm = max(np.max(np.abs(grad_U)), np.max(np.abs(grad_V)), np.max(np.abs(grad_W)))
            if grad_norm < 1e-12:
                break
            
            # Scale learning rate to avoid overshooting
            scale = min(1.0, 1.0 / (grad_norm + 1e-8))
            eff_lr = lr * scale
            
            U -= eff_lr * grad_U
            V -= eff_lr * grad_V
            W -= eff_lr * grad_W
            
            lr *= lr_decay
        
        return best_U, best_V, best_W, best_obj

    def try_rank(target_rank, num_attempts=10, seed_base=0):
        """Try to find a valid decomposition at the given rank."""
        for attempt in range(num_attempts):
            np.random.seed(seed_base + attempt * 1000 + target_rank)
            
            # Initialize near schoolbook structure with random perturbations
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            # Also try structured initialization based on partial schoolbook
            if attempt % 2 == 0:
                # Use some schoolbook terms as anchor
                num_anchor = min(target_rank // 2, n * m * p)
                for c in range(num_anchor):
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                # Randomize the rest
                for c in range(num_anchor, target_rank):
                    U[:, c] = np.random.randn(n * m) * 0.2
                    V[:, c] = np.random.randn(m * p) * 0.2
                    W[:, c] = np.random.randn(p * n) * 0.2
            
            U, V, W, obj = gradient_descent(U, V, W, max_steps=1500)
            
            if check_decomposition(U, V, W, target_rank):
                return U, V, W, True
        
        return None, None, None, False

    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s limit
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    # Try lower ranks first, starting from known theoretical bounds
    # For 3x3x8, the rank is at most 72 (schoolbook)
    # Known results suggest rank could be significantly lower
    # Try ranks in decreasing order of likelihood
    
    target_ranks = list(range(50, 73))  # Try from 50 to 72
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Use more attempts for lower ranks (harder to find)
        num_attempts = 5 if target_rank < 60 else 3
        
        U, V, W, success = try_rank(target_rank, num_attempts=num_attempts)
        
        if success:
            best_rank = target_rank
            best_U = U
            best_V = V
            best_W = W
            # Continue searching for even lower ranks
    
    # If we didn't find anything better, use schoolbook
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
