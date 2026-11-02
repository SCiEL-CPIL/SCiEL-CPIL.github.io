"""Matrix multiplication 2x3x9 rank-minimal decomposition using Kronecker structure."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    Strategy: Exploit the structure 9 = 3*3 by viewing the 3x9 matrix as a 3x3 block matrix
    where each block is 1x3. We use the fact that the 3x9 multiplication can be decomposed
    using the Kronecker product structure.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 27 upward with structured approaches first
    
    # First, try the structured Kronecker-based decomposition
    for rank in [27, 30, 36, 45, 54]:
        if time.time() > deadline - 5:
            break
        
        found = _try_structured_decompose(n, m, p, rank, deadline)
        if found is not None:
            return found
    
    # Fallback: schoolbook decomposition with rank 54
    return _schoolbook(n, m, p)

def _try_structured_decompose(n, m, p, rank, deadline):
    """Try to find a structured decomposition using Kronecker product structure."""
    # Since p = 9 = 3*3, we can exploit this structure
    # The 3x9 multiplication can be seen as 3 separate 3x3 multiplications
    # if we reshape the 9 columns into 3 groups of 3
    
    # For each block, we need a 2x3x3 decomposition (rank 6)
    # Total: 3 blocks * 6 = 18, but we can potentially share structure
    
    # Let's try a direct approach: initialize with structured patterns and optimize
    np.random.seed(42)
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple structured initializations
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + trial * 13)
        
        # Structured initialization based on Kronecker products
        U, V, W = _initialize_structured(n, m, p, rank, trial)
        
        # Optimize using accelerated gradient descent
        result = _optimize_fast(U, V, W, n, m, p, deadline, max_iter=200)
        if result is not None:
            res = _compute_residual_fast(result[0], result[1], result[2], n, m, p)
            residual_norm = np.max(np.abs(res))
            if residual_norm < 1e-10:
                return _finalize(result[0], result[1], result[2], n, m, p)
            if residual_norm < best_residual:
                best_residual = residual_norm
                best_U, best_V, best_W = result[0].copy(), result[1].copy(), result[2].copy()
        
        if best_residual < 1e-8:
            return _finalize(best_U, best_V, best_W, n, m, p)
    
    return None

def _initialize_structured(n, m, p, rank, trial):
    """Initialize with structured patterns based on Kronecker products."""
    # Create base patterns
    if trial % 3 == 0:
        # Kronecker-based initialization
        U = np.random.randn(n * m, rank) * 0.3
        V = np.random.randn(m * p, rank) * 0.3
        W = np.random.randn(p * n, rank) * 0.3
    elif trial % 3 == 1:
        # Block-structured initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Divide rank into blocks
        block_size = max(1, rank // 3)
        for block in range(3):
            start = block * block_size
            end = min((block + 1) * block_size, rank)
            for i in range(n):
                for j in range(m):
                    U[i * m + j, start:end] = np.random.randn(end - start) * 0.5
            for j in range(m):
                for k in range(p):
                    V[j * p + k, start:end] = np.random.randn(end - start) * 0.5
            for k in range(p):
                for i in range(n):
                    W[k * n + i, start:end] = np.random.randn(end - start) * 0.5
    else:
        # Sparse initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            # Randomly select entries to be nonzero
            for _ in range(min(n * m, 3)):
                idx = np.random.randint(0, n * m)
                U[idx, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            for _ in range(min(m * p, 5)):
                idx = np.random.randint(0, m * p)
                V[idx, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            for _ in range(min(p * n, 4)):
                idx = np.random.randint(0, p * n)
                W[idx, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
    
    return U, V, W

def _optimize_fast(U, V, W, n, m, p, deadline, max_iter=200):
    """Fast optimization using momentum and adaptive learning rate."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Momentum buffers
    mU = np.zeros_like(U)
    mV = np.zeros_like(V)
    mW = np.zeros_like(W)
    
    lr = 0.05
    momentum = 0.9
    best_residual = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Compute residual
        res = _compute_residual_fast(U, V, W, n, m, p)
        residual_norm = np.max(np.abs(res))
        
        if residual_norm < 1e-10:
            return U, V, W
        
        if residual_norm < best_residual:
            best_residual = residual_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = res[i, j, k]
                    dU[i * m + j, :] += r * V[j * p + k, :] * W[k * n + i, :]
                    dV[j * p + k, :] += r * U[i * m + j, :] * W[k * n + i, :]
                    dW[k * n + i, :] += r * U[i * m + j, :] * V[j * p + k, :]
        
        # Update with momentum
        mU = momentum * mU + (1 - momentum) * dU
        mV = momentum * mV + (1 - momentum) * dV
        mW = momentum * mW + (1 - momentum) * dW
        
        U -= lr * mU
        V -= lr * mV
        W -= lr * mW
        
        # Decay learning rate
        lr *= 0.995
        
        if best_residual < 1e-8:
            return best_U, best_V, best_W
    
    return best_U, best_V, best_W

def _compute_residual_fast(U, V, W, n, m, p):
    """Compute the residual tensor using matrix operations."""
    res = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]) - 1.0
    return res

def _finalize(U, V, W, n, m, p):
    """Finalize and verify the decomposition."""
    # Verify
    res = _compute_residual_fast(U, V, W, n, m, p)
    if np.max(np.abs(res)) < 1e-8:
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        res_rounded = _compute_residual_fast(U_rounded, V_rounded, W_rounded, n, m, p)
        if np.max(np.abs(res_rounded)) < 1e-6:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
