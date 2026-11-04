"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _structured_init(n, m, p, rank, seed=None):
    """Create structured initialization based on block structure."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Initialize with small random values to break symmetry
    U += np.random.randn(n * m, rank) * 0.01
    V += np.random.randn(m * p, rank) * 0.01
    W += np.random.randn(p * n, rank) * 0.01
    
    return U, V, W

def _try_rank_optimized(n, m, p, rank, max_iter=200, seed=None, lr_schedule='cosine'):
    """Try to find a rank-`rank` decomposition via optimized gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    
    U, V, W = _structured_init(n, m, p, rank, seed)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Adaptive learning rate
        if lr_schedule == 'cosine':
            lr = 0.05 * (1 + np.cos(np.pi * it / max_iter))
        else:
            lr = 0.05
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += diff * diff
                    dU[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
        
        # Gradient step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None

def _try_block_decomposition(n, m, p, rank, seed=None):
    """Try block-structured decomposition."""
    if seed is not None:
        np.random.seed(seed)
    
    # Try to use the fact that we can decompose into smaller blocks
    # For 3x5x6, try splitting into 3x3x3 + 3x2x3 + 3x3x3 blocks
    # This is a heuristic approach
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Initialize with random values
    U += np.random.randn(n * m, rank) * 0.1
    V += np.random.randn(m * p, rank) * 0.1
    W += np.random.randn(p * n, rank) * 0.1
    
    # Optimize
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(100):
        lr = 0.02
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += diff * diff
                    dU[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None

def _polish_decomposition(U, V, W, n, m, p, max_iter=500, tol=1e-14):
    """Polish a near-exact decomposition using high-precision optimization."""
    rank = U.shape[1]
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    for it in range(max_iter):
        # Compute current error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (1.0 - s) ** 2
        if err < tol:
            break
        
        # Use smaller learning rate for polishing
        lr = 0.01
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    dU[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Final check
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (1.0 - s) ** 2
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook (rank = 90)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Focus on ranks that are likely to be achievable
    # Theoretical lower bound is max(n*m, m*p, p*n) = 30
    # But practical lower bounds are higher
    
    # Try a range of ranks, starting from more promising ones
    ranks_to_try = list(range(85, best_rank)) + list(range(70, 85)) + list(range(50, 70))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple approaches for each rank
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            seed = rank * 1000 + attempt * 7
            
            # Try optimized random initialization
            result = _try_rank_optimized(n, m, p, rank, max_iter=150, seed=seed)
            if result is not None and _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    # If we have time, try to polish the best result to ensure exactness
    if time.time() < deadline - 3:
        polished = _polish_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p)
        if polished is not None:
            best_result = polished
    
    # Convert to lists
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
# EVOLVE-BLOCK-END
