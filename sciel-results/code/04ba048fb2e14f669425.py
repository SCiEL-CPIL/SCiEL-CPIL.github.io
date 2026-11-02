"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Uses QCQP formulation with gradient-based optimization to find low-rank decompositions.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 30
    dim_w = p * n  # 12
    
    # Build the target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition requires: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    def schoolbook():
        """Return the trivial schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def check_exact(U, V, W, n, m, p, tol=1e-8):
        """
        Check if U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        For all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        """
        r = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(2 * x) / 2.0
    
    def try_qcq_gradient(rank, max_iter=5000, lr_schedule=True, seed=42):
        """
        Use gradient-based optimization to find a rank-r decomposition.
        We minimize the sum of squared errors:
        L = sum_{i,j,k} (1 - sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t])^2
        """
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        # Precompute indices for efficiency
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    indices.append((u_idx, v_idx, w_idx))
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        momentum = 0.9
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        
        for it in range(max_iter):
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            loss = 0.0
            
            for u_idx, v_idx, w_idx in indices:
                # Compute the inner sum: sum_t U[u_idx,t] * V[v_idx,t] * W[w_idx,t]
                inner = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                error = inner - 1.0
                loss += error * error
                
                # Gradient with respect to U[u_idx, t]:
                # d/dU[u_idx,t] (error^2) = 2 * error * V[v_idx,t] * W[w_idx,t]
                # Similarly for V and W
                grad_U[u_idx] += 2 * error * V[v_idx] * W[w_idx]
                grad_V[v_idx] += 2 * error * U[u_idx] * W[w_idx]
                grad_W[w_idx] += 2 * error * U[u_idx] * V[v_idx]
            
            # Normalize gradients
            grad_U /= len(indices)
            grad_V /= len(indices)
            grad_W /= len(indices)
            
            # Update with momentum
            v_U = momentum * v_U + lr * grad_U
            v_V = momentum * v_V + lr * grad_V
            v_W = momentum * v_W + lr * grad_W
            
            U -= v_U
            V -= v_V
            W -= v_W
            
            # Track best solution
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate
            if lr_schedule:
                lr = 0.01 / (1 + 0.001 * it)
            
            # Early stopping if loss is very small
            if best_loss < 1e-10:
                break
            
            # Check if we've converged
            if it % 100 == 0 and best_loss < 1e-12:
                break
        
        # Round to half-integers and check exactness
        U_rounded = round_to_half(best_U)
        V_rounded = round_to_half(best_V)
        W_rounded = round_to_half(best_W)
        
        if check_exact(U_rounded, V_rounded, W_rounded, n, m, p):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        
        # Also try without rounding (in case exact solution has non-half-integer values)
        if check_exact(best_U, best_V, best_W, n, m, p):
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
        
        return None
    
    def try_structured_decomposition(rank, seed=42):
        """
        Try structured decompositions using block patterns.
        For 2x5 * 5x6, we can exploit the structure.
        """
        np.random.seed(seed)
        
        # Try different structured initializations
        for strategy in range(5):
            if strategy == 0:
                # Block diagonal structure
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                
                # Try to use rank-1 tensors in a structured way
                for t in range(rank):
                    # Assign each t to a specific (i,j,k) pattern
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n * m)) % p
                    
                    U[i * m + j, t] = 1
                    V[j * p + k, t] = 1
                    W[k * n + i, t] = 1
                    
                    # Add some random perturbation
                    if t < rank // 2:
                        U[i * m + j, t] += np.random.randn() * 0.5
                        V[j * p + k, t] += np.random.randn() * 0.5
                        W[k * n + i, t] += np.random.randn() * 0.5
            
            elif strategy == 1:
                # Random sparse initialization
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                
                for t in range(rank):
                    # Randomly select a few non-zero entries
                    num_nonzero_u = max(1, dim_u // 3)
                    num_nonzero_v = max(1, dim_v // 3)
                    num_nonzero_w = max(1, dim_w // 3)
                    
                    u_indices = np.random.choice(dim_u, size=num_nonzero_u, replace=False)
                    v_indices = np.random.choice(dim_v, size=num_nonzero_v, replace=False)
                    w_indices = np.random.choice(dim_w, size=num_nonzero_w, replace=False)
                    
                    U[u_indices, t] = np.random.choice([-1, 1], size=num_nonzero_u)
                    V[v_indices, t] = np.random.choice([-1, 1], size=num_nonzero_v)
                    W[w_indices, t] = np.random.choice([-1, 1], size=num_nonzero_w)
            
            elif strategy == 2:
                # Use Kronecker product structure
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                
                # Try to decompose using tensor products of smaller matrices
                for t in range(rank):
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n * m)) % p
                    
                    U[i * m + j, t] = 1.0
                    V[j * p + k, t] = 1.0
                    W[k * n + i, t] = 1.0
                    
                    # Add cross terms
                    if t < rank // 2 and j < m - 1:
                        U[i * m + j + 1, t] = 0.5
                        V[(j + 1) * p + k, t] = 0.5
                        W[k * n + i, t] = 0.5
            
            elif strategy == 3:
                # Fourier-like structure
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                
                for t in range(rank):
                    phase = 2 * np.pi * t / rank
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, t] = np.cos(phase * (i + j))
                    
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, t] = np.sin(phase * (j + k))
                    
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i, t] = np.cos(phase * (k + i))
            
            elif strategy == 4:
                # Random dense initialization
                U = np.random.randn(dim_u, rank) * 0.5
                V = np.random.randn(dim_v, rank) * 0.5
                W = np.random.randn(dim_w, rank) * 0.5
            
            # Refine using gradient descent
            result = try_qcq_gradient(rank, max_iter=3000, seed=seed + strategy)
            if result is not None:
                return result
        
        return None
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But we can try lower ranks with structured approaches
    
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    # First, try to find a decomposition with rank < 60
    # Start from rank 30 and go up
    
    best_result = None
    best_rank = n * m * p  # 60
    
    # Try structured approaches for various ranks
    for rank in range(30, best_rank + 1, 2):
        if time.time() - start_time > time_limit:
            break
        
        # Try gradient-based approach
        result = try_qcq_gradient(rank, max_iter=2000, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try structured approaches
        result = try_structured_decomposition(rank, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # If we didn't find anything better, try a few more random seeds
    if best_rank == n * m * p:
        for rank in range(35, 55, 3):
            if time.time() - start_time > time_limit:
                break
            
            for seed in [100 + rank, 200 + rank, 300 + rank]:
                result = try_qcq_gradient(rank, max_iter=1500, seed=seed)
                if result is not None:
                    best_result = result
                    best_rank = rank
                    break
            if best_rank < n * m * p:
                break
    
    # Return the best result found, or schoolbook as fallback
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
