"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Approach: Start with schoolbook (rank 90) and try to reduce rank by finding linear dependencies
    # or use a randomized search for lower ranks.
    
    # First, generate the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Try to find a lower rank decomposition using randomized search with pruning
    # We search for ranks from 80 to 89
    for target_rank in range(80, 90):
        result = _randomized_search(n, m, p, target_rank)
        if result is not None:
            return result
            
    # If no lower rank found, return schoolbook
    return _to_list(U, V, W)

def _randomized_search(n, m, p, rank, max_iter=5000, seed=42):
    nm, mp, pn = n * m, m * p, p * n
    np.random.seed(seed)
    
    # Initialize with random sparse vectors
    # Use a sparsity pattern to help convergence
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Randomly assign non-zero entries
    for r in range(rank):
        # Each term should contribute to a subset of (i,j,k)
        # Randomly pick indices
        idx_u = np.random.choice(nm, size=3, replace=False)
        idx_v = np.random.choice(mp, size=3, replace=False)
        idx_w = np.random.choice(pn, size=3, replace=False)
        
        U[idx_u, r] = 1
        V[idx_v, r] = 1
        W[idx_w, r] = 1
        
    # Gradient descent with rounding
    lr = 0.05
    prev_loss = np.inf
    
    for it in range(max_iter):
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j]
                    vk = V[j * p + k]
                    wk = W[k * n + i]
                    val = np.sum(ui * vk * wk)
                    err = val - 1.0
                    loss += err * err
                    grad_U[i * m + j] += 2 * err * vk * wk
                    grad_V[j * p + k] += 2 * err * ui * wk
                    grad_W[k * n + i] += 2 * err * ui * vk
        
        if loss < 1e-8:
            # Check if we can round to integers/half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if _check_decomposition(n, m, p, U_rounded, V_rounded, W_rounded):
                return _to_list(U_rounded, V_rounded, W_rounded)
            
        if abs(prev_loss - loss) < 1e-12 and it > 100:
            break
        prev_loss = loss
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Enforce sparsity by zeroing out small values
        U[np.abs(U) < 0.1] = 0
        V[np.abs(V) < 0.1] = 0
        W[np.abs(W) < 0.1] = 0
        
    return None

def _check_decomposition(n, m, p, U, V, W):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _to_list(U, V, W):
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
