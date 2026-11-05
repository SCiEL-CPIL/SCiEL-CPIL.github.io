"""
Task: Matrix multiplication 3,3,7
Find a rank-minimal exact tensor decomposition.
We search for a lower-rank decomposition using numerical methods,
falling back to schoolbook (rank 63) if no improvement is found.
"""

import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 63
    
    # Build the target tensor nonzero indices
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    # Schoolbook decomposition as fallback
    rank = schoolbook_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Helper to check if a decomposition is valid
    def check_valid(U, V, W, tol=1e-4):
        # Check nonzero entries
        for (a, b, c) in nonzero_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check a sample of zero entries
        np.random.seed(12345)
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            # Check if this is a nonzero entry
            is_nonzero = any(a == na and b == nb and c == nc for (na, nb, nc) in nonzero_indices)
            if not is_nonzero:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    # Try to find lower-rank decompositions using structured search
    # Strategy: start from schoolbook and try to merge terms
    
    # Alternative: use a genetic algorithm or simulated annealing
    # For now, try random initialization with Adam optimizer
    
    def optimize(U, V, W, lr=0.001, max_iter=1000):
        """Simple gradient descent with adaptive learning rate."""
        m1_U = np.zeros_like(U); m2_U = np.zeros_like(U)
        m1_V = np.zeros_like(V); m2_V = np.zeros_like(V)
        m1_W = np.zeros_like(W); m2_W = np.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            t += 1
            
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            # Nonzero constraints
            for (a, b, c) in nonzero_indices:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                residual = 1.0 - val
                U_grad[a, :] += residual * V[b, :] * W[c, :]
                V_grad[b, :] += residual * U[a, :] * W[c, :]
                W_grad[c, :] += residual * U[a, :] * V[b, :]
            
            # Zero constraints (sample)
            for _ in range(20):
                a = np.random.randint(0, n * m)
                b = np.random.randint(0, m * p)
                c = np.random.randint(0, p * n)
                is_nonzero = any(a == na and b == nb and c == nc for (na, nb, nc) in nonzero_indices)
                if not is_nonzero:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    U_grad[a, :] += val * V[b, :] * W[c, :]
                    V_grad[b, :] += val * U[a, :] * W[c, :]
                    W_grad[c, :] += val * U[a, :] * V[b, :]
            
            # Adam update
            m1_U = beta1 * m1_U + (1 - beta1) * U_grad
            m2_U = beta2 * m2_U + (1 - beta2) * U_grad**2
            U -= lr * (m1_U / (1 - beta1**t)) / (np.sqrt(m2_U / (1 - beta2**t)) + eps)
            
            m1_V = beta1 * m1_V + (1 - beta1) * V_grad
            m2_V = beta2 * m2_V + (1 - beta2) * V_grad**2
            V -= lr * (m1_V / (1 - beta1**t)) / (np.sqrt(m2_V / (1 - beta2**t)) + eps)
            
            m1_W = beta1 * m1_W + (1 - beta1) * W_grad
            m2_W = beta2 * m2_W + (1 - beta2) * W_grad**2
            W -= lr * (m1_W / (1 - beta1**t)) / (np.sqrt(m2_W / (1 - beta2**t)) + eps)
            
            if iteration % 100 == 99:
                if check_valid(U, V, W, tol=1e-3):
                    return U, V, W, True
        
        return U, V, W, False
    
    # Try various ranks
    for r in [56, 54, 52, 50, 48, 46, 44, 42, 40]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + r * 100 + attempt)
            U_try = np.random.randn(n * m, r) * 0.5
            V_try = np.random.randn(m * p, r) * 0.5
            W_try = np.random.randn(p * n, r) * 0.5
            
            U_try, V_try, W_try, converged = optimize(U_try, V_try, W_try, lr=0.005, max_iter=500)
            
            if converged and check_valid(U_try, V_try, W_try, tol=1e-4):
                rank = r
                U, V, W = U_try, V_try, W_try
                break
        
        if rank < schoolbook_rank:
            break
    
    # Convert to lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
