"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _compute_val(U, V, W, n, m, p):
    """Compute val[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] using vectorized ops"""
    val = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            # U[i*m+j] is shape (rank,), V[j*p + k] for all k, W[k*n+i] for all k
            # For fixed i,j: val[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
            # Vectorize over k: for each k, compute dot product
            u_row = U[i * m + j]  # shape (rank,)
            for k in range(p):
                val[i, j, k] = np.dot(u_row, V[j * p + k] * W[k * n + i])
    return val

def _compute_gradients(U, V, W, n, m, p):
    """Compute analytic gradients of the error function using vectorized operations"""
    val = _compute_val(U, V, W, n, m, p)
    diff = val - 1.0
    rank = U.shape[1]
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                d = 2.0 * diff[i, j, k]
                grad_U[i * m + j] += d * V[j * p + k] * W[k * n + i]
                grad_V[j * p + k] += d * U[i * m + j] * W[k * n + i]
                grad_W[k * n + i] += d * U[i * m + j] * V[j * p + k]
    
    return grad_U, grad_V, grad_W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact within tolerance"""
    val = _compute_val(U, V, W, n, m, p)
    return np.max(np.abs(val - 1.0)) < tol

def _smooth_loss(U, V, W, n, m, p, beta=100.0):
    """Smooth surrogate for max-norm loss using log-sum-exp"""
    val = _compute_val(U, V, W, n, m, p)
    diff = val - 1.0
    abs_diff = np.abs(diff)
    # Log-sum-exp for smooth max
    max_val = np.max(abs_diff)
    if max_val < 1e-10:
        return max_val
    shifted = abs_diff / beta
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / beta
    return beta * log_sum

def _smooth_gradients(U, V, W, n, m, p, beta=100.0):
    """Gradients of the smooth log-sum-exp loss"""
    val = _compute_val(U, V, W, n, m, p)
    diff = val - 1.0
    abs_diff = np.abs(diff)
    rank = U.shape[1]
    
    max_val = np.max(abs_diff)
    if max_val < 1e-10:
        return np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    
    shifted = abs_diff / beta
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # softmax weights
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sign = 1.0 if diff[i, j, k] >= 0 else -1.0
                w = 2.0 * weights[i, j, k] * sign
                grad_U[i * m + j] += w * V[j * p + k] * W[k * n + i]
                grad_V[j * p + k] += w * U[i * m + j] * W[k * n + i]
                grad_W[k * n + i] += w * U[i * m + j] * V[j * p + k]
    
    return grad_U, grad_V, grad_W

def _adam_update(params, grads, m_mom, v_mom, t, lr, beta1, beta2, eps):
    """Adam update step"""
    for i, (p, g) in enumerate(zip(params, grads)):
        m_mom[i] = beta1 * m_mom[i] + (1 - beta1) * g
        v_mom[i] = beta2 * v_mom[i] + (1 - beta2) * g ** 2
        m_hat = m_mom[i] / (1 - beta1 ** t)
        v_hat = v_mom[i] / (1 - beta2 ** t)
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition using structured initialization and optimization."""
    np.random.seed(42 + rank)
    
    # Try multiple initializations
    for trial in range(6):
        if time.time() > deadline:
            return None
        
        # Different initialization strategies
        if trial == 0:
            # Pure random
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
        elif trial == 1:
            # Structured: group by k
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            U += np.random.randn(n * m, rank) * 0.1
            V += np.random.randn(m * p, rank) * 0.1
            W += np.random.randn(p * n, rank) * 0.1
        elif trial == 2:
            # Block structure: exploit the 4x9 aspect ratio
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            cols_per_block = rank // p
            for k in range(p):
                start = k * cols_per_block
                end = start + cols_per_block
                for c in range(start, min(end, rank)):
                    i = (c - start) % n
                    j = ((c - start) // n) % m
                    if i * m + j < n * m:
                        U[i * m + j, c] = 1.0
                    if j * p + k < m * p:
                        V[j * p + k, c] = 1.0
                    if k * n + i < p * n:
                        W[k * n + i, c] = 1.0
            U += np.random.randn(n * m, rank) * 0.05
            V += np.random.randn(m * p, rank) * 0.05
            W += np.random.randn(p * n, rank) * 0.05
        elif trial == 3:
            # Sparse random
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            mask = np.random.rand(n * m, rank) < 0.3
            U[mask] = 0
            mask = np.random.rand(m * p, rank) < 0.3
            V[mask] = 0
            mask = np.random.rand(p * n, rank) < 0.3
            W[mask] = 0
        elif trial == 4:
            # Cyclic structure
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for c in range(rank):
                i = (c * 7) % n
                j = (c * 13) % m
                k = (c * 19) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            U += np.random.randn(n * m, rank) * 0.2
            V += np.random.randn(m * p, rank) * 0.2
            W += np.random.randn(p * n, rank) * 0.2
        else:
            # Schoolbook-like with random perturbations on a subset of terms
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            # Use first min(rank, n*m*p) terms as schoolbook
            num_terms = min(rank, n * m * p)
            for c in range(num_terms):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            # Add random structure for remaining columns
            for c in range(num_terms, rank):
                U[np.random.randint(n * m), c] = 1.0
                V[np.random.randint(m * p), c] = 1.0
                W[np.random.randint(p * n), c] = 1.0
            U += np.random.randn(n * m, rank) * 0.05
            V += np.random.randn(m * p, rank) * 0.05
            W += np.random.randn(p * n, rank) * 0.05
        
        # Check if already valid
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return (U, V, W)
        
        # Adam optimizer with smooth loss
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        params = [U, V, W]
        m_mom = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
        v_mom = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
        
        max_steps = 200
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Compute gradients using smooth loss
            grad_U, grad_V, grad_W = _smooth_gradients(U, V, W, n, m, p, beta=50.0)
            grads = [grad_U, grad_V, grad_W]
            
            # Update Adam moments and parameters
            t = step + 1
            _adam_update(params, grads, m_mom, v_mom, t, lr, beta1, beta2, eps)
            U, V, W = params
            
            # Track best using actual max-norm loss
            if step % 10 == 0 or step == max_steps - 1:
                val = _compute_val(U, V, W, n, m, p)
                loss = np.max(np.abs(val - 1.0))
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-6:
                    return (U, V, W)
        
        # Check best found
        if best_loss < 1e-6:
            return (best_U, best_V, best_W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit, keep 5s safety margin
    
    # Generate schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 36
    # Try strategic ranks - start from lower and go up, or use smart ordering
    candidate_ranks = list(range(60, 35, -1)) + list(range(100, 60, -5))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        # Try to find a decomposition of this rank
        result = _try_rank(n, m, p, target_rank, deadline - 10)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
