<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_rank(n, m, p, rank, max_iters=3000, seed=42):
    """Try to find a rank-`rank` decomposition via random init + local search."""
    np.random.seed(seed)
    random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to make products of order 1
    U = U / (rank ** 0.5)
    V = V / (rank ** 0.5)
    W = W / (rank ** 0.5)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    for it in range(max_iters):
        # Compute error and gradients efficiently
        err = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    err += (s - 1.0) ** 2
                    grad = (s - 1.0)
                    dU[i * m + j] += grad * V[j * p + k] * W[k * n + i]
                    dV[j * p + k] += grad * U[i * m + j] * W[k * n + i]
                    dW[k * n + i] += grad * U[i * m + j] * V[j * p + k]
        
        err = np.sqrt(err / (n * m * p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            break
        
        # Gradient step with adaptive learning rate
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 300 == 299:
            lr *= 0.7
    
    if best_err < 1e-8:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    return None

def _try_integer_rank(n, m, p, rank, max_trials=2000, seed=42):
    """Try to find a rank-`rank` decomposition with small integer entries."""
    np.random.seed(seed)
    random.seed(seed)
    
    vals = [0, 1, -1, 2, -2, 3, -3, 4, -4]
    
    for t in range(max_trials):
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None

def _try_kronecker_structured(n, m, p, rank, max_iters=2000, seed=42):
    """
    Try a structured approach using Kronecker-like patterns.
    We initialize with structured patterns based on binary representations
    and then refine with gradient descent.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize with structured patterns
    # Use bits of indices to create structure
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each column, assign a pattern based on the column index bits
    for c in range(rank):
        # Create a pseudo-random but structured pattern
        bit_pattern = 1 << (c % 16)
        
        # U: pattern based on i, j
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # Simple hash-like structure
                val = (idx + c) % 5 - 2  # Range -2 to 2
                U[idx, c] = val if (bit_pattern & 1) else 0
                bit_pattern >>= 1
        
        # V: pattern based on j, k
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                val = (idx + c) % 5 - 2
                V[idx, c] = val if (bit_pattern & 1) else 0
                bit_pattern >>= 1
        
        # W: pattern based on k, i
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                val = (idx + c) % 5 - 2
                W[idx, c] = val if (bit_pattern & 1) else 0
                bit_pattern >>= 1
    
    # Normalize
    U /= (rank ** 0.5)
    V /= (rank ** 0.5)
    W /= (rank ** 0.5)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.03
    for it in range(max_iters):
        # Compute error and gradients
        err = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    err += (s - 1.0) ** 2
                    grad = (s - 1.0)
                    dU[i * m + j] += grad * V[j * p + k] * W[k * n + i]
                    dV[j * p + k] += grad * U[i * m + j] * W[k * n + i]
                    dW[k * n + i] += grad * U[i * m + j] * V[j * p + k]
        
        err = np.sqrt(err / (n * m * p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            break
        
        # Gradient step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 400 == 399:
            lr *= 0.7
    
    if best_err < 1e-8:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Try ranks from 50 down to 30
    for rank in range(50, 29, -1):
        if time.time() > deadline - 15:
            break
        
        found = None
        
        # First try integer search (fast, might find exact small-integer solutions)
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            result = _try_integer_rank(n, m, p, rank, max_trials=500, seed=seed)
            if result is not None:
                found = result
                break
        
        # If not found, try random real-valued search
        if found is None:
            for seed in range(5):
                if time.time() > deadline - 15:
                    break
                result = _try_random_rank(n, m, p, rank, max_iters=2500, seed=seed)
                if result is not None:
                    found = result
                    break
        
        # If not found, try structured Kronecker-like search
        if found is None:
            for seed in range(3):
                if time.time() > deadline - 15:
                    break
                result = _try_kronecker_structured(n, m, p, rank, max_iters=2000, seed=seed)
                if result is not None:
                    found = result
                    break
        
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
                break  # Found the lowest rank we tried
    
    return best
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _compute_error_and_grads(U, V, W, n, m, p):
    """Vectorized computation of error and gradients."""
    rank = U.shape[1]
    
    # Build index arrays for efficient access
    # U has shape (n*m, rank), indexed by i*m+j
    # V has shape (m*p, rank), indexed by j*p+k
    # W has shape (p*n, rank), indexed by k*n+i
    
    # Precompute the outer products for each (i,j,k)
    # For each c, we need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Vectorized approach: for each column c, compute the tensor product
    err = 0.0
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    # For each rank component
    for c in range(rank):
        u_c = U[:, c]  # shape (n*m,)
        v_c = V[:, c]  # shape (m*p,)
        w_c = W[:, c]  # shape (p*n,)
        
        # Compute the contribution of this rank component to each (i,j,k)
        # We need to compute the tensor T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        
        # Reshape for broadcasting
        u_reshaped = u_c.reshape(n, m)
        v_reshaped = v_c.reshape(m, p)
        w_reshaped = w_c.reshape(p, n)
        
        # Compute the rank-1 contribution: T_c[i,j,k] = u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        # This is an outer product of u_reshaped[i,j], v_reshaped[j,k], w_reshaped[k,i]
        
        # Use einsum for efficiency
        # T_c[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
        T_c = np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped)
        
        # Add to total tensor
        # The target tensor has all entries = 1
        # Current partial sum from all previous columns + this column
        
    # Alternative: compute the full tensor sum first, then gradients
    # Let S[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want S[i,j,k] = 1 for all i,j,k
    
    # Build S
    S = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        S += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
    
    # Error
    err = np.sum((S - 1.0) ** 2)
    
    # Gradients: dU[i*m+j, c] = sum_{i',j',k'} (S[i',j',k'] - 1) * delta(i',i) * delta(j',j) * V[j'*p+k', c] * W[k'*n+i', c]
    # = sum_{k'} (S[i,j,k'] - 1) * V[j*p+k', c] * W[k'*n+i, c]
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        
        # dU[i,j,c] = sum_k (S[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        dU_c = np.einsum('ijk,jk,ki->ij', S - 1.0, v_c, w_c)
        dU[:, c] = dU_c.flatten()
        
        # dV[j,k,c] = sum_i (S[i,j,k] - 1) * U[i,j,c] * W[k,i,c]
        dV_c = np.einsum('ijk,ij,ki->jk', S - 1.0, u_c, w_c)
        dV[:, c] = dV_c.flatten()
        
        # dW[k,i,c] = sum_j (S[i,j,k] - 1) * U[i,j,c] * V[j,k,c]
        dW_c = np.einsum('ijk,ij,jk->ki', S - 1.0, u_c, v_c)
        dW[:, c] = dW_c.flatten()
    
    return err, dU, dV, dW

def _try_random_rank(n, m, p, rank, max_iters=5000, seed=42, init_method='random'):
    """Try to find a rank-`rank` decomposition via random init + local search."""
    np.random.seed(seed)
    random.seed(seed)
    
    if init_method == 'random':
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        scale = (rank ** 0.5)
        U /= scale
        V /= scale
        W /= scale
    elif init_method == 'structured':
        # Use a more structured initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            # Create diverse patterns
            U[:, c] = np.random.randn(n * m) * 0.5
            V[:, c] = np.random.randn(m * p) * 0.5
            W[:, c] = np.random.randn(p * n) * 0.5
        
        U /= (rank ** 0.5)
        V /= (rank ** 0.5)
        W /= (rank ** 0.5)
    else:  # 'binary'
        U = np.random.randint(0, 2, (n * m, rank)).astype(float) - 0.5
        V = np.random.randint(0, 2, (m * p, rank)).astype(float) - 0.5
        W = np.random.randint(0, 2, (p * n, rank)).astype(float) - 0.5
        U /= (rank ** 0.5)
        V /= (rank ** 0.5)
        W /= (rank ** 0.5)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.1
    momentum = 0.9
    v_U = np.zeros_like(U)
    v_V = np.zeros_like(V)
    v_W = np.zeros_like(W)
    
    for it in range(max_iters):
        err, dU, dV, dW = _compute_error_and_grads(U, V, W, n, m, p)
        err = np.sqrt(err / (n * m * p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Adam-like update
        v_U = momentum * v_U + (1 - momentum) * dU
        v_V = momentum * v_V + (1 - momentum) * dV
        v_W = momentum * v_W + (1 - momentum) * dW
        
        U -= lr * v_U
        V -= lr * v_V
        W -= lr * v_W
        
        # Decay learning rate
        if it % 500 == 499:
            lr *= 0.8
    
    if best_err < 1e-8:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    return None

def _try_integer_rank(n, m, p, rank, max_trials=3000, seed=42):
    """Try to find a rank-`rank` decomposition with small integer entries."""
    np.random.seed(seed)
    random.seed(seed)
    
    vals = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    
    for t in range(max_trials):
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Try ranks from 45 down to 30
    for rank in range(45, 29, -1):
        if time.time() > deadline - 20:
            break
        
        found = None
        
        # First try integer search (fast, might find exact small-integer solutions)
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            result = _try_integer_rank(n, m, p, rank, max_trials=800, seed=seed)
            if result is not None:
                found = result
                break
        
        # If not found, try random real-valued search with different initializations
        if found is None:
            init_methods = ['random', 'structured', 'binary']
            for seed in range(8):
                if time.time() > deadline - 20:
                    break
                init = init_methods[seed % len(init_methods)]
                result = _try_random_rank(n, m, p, rank, max_iters=3000, seed=seed, init_method=init)
                if result is not None:
                    found = result
                    break
        
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
                break  # Found the lowest rank we tried
    
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
