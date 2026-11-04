"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 115.0  # 120s limit with 5s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify(U, V, W, n=4, m=5, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    except Exception:
        return False
        
    if np.max(np.abs(T - 1.0)) > 1e-8:
        return False
    return True

def _schoolbook(n=4, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_winograd(n, m, p, r, seed=42):
    """
    Try a structured ansatz inspired by Winograd/FFT methods.
    Use circulant-like patterns to reduce effective rank.
    """
    np.random.seed(seed)
    
    # Initialize with structured patterns
    U = np.zeros((n, m, r))
    V = np.zeros((m, p, r))
    W = np.zeros((p, n, r))
    
    # Strategy: use block structure
    # For each "frequency" component c, assign patterns
    for c in range(r):
        # Random but structured initialization
        phase = np.random.uniform(0, 2*np.pi)
        
        # U pattern: depend on i and j
        for i in range(n):
            for j in range(m):
                U[i, j, c] = np.cos(phase * (i + j))
        
        # V pattern: depend on j and k  
        for j in range(m):
            for k in range(p):
                V[j, k, c] = np.cos(phase * (j - k))
        
        # W pattern: depend on k and i
        for k in range(p):
            for i in range(n):
                W[k, i, c] = np.cos(phase * (k + i))
    
    # Now do local optimization
    best_U, best_V, best_W, best_err = _local_optimize(U.copy(), V.copy(), W.copy(), n, m, p, steps=200)
    
    return best_U, best_V, best_W, best_err

def _local_optimize(U, V, W, n, m, p, steps=200, lr=0.01):
    """
    Adam-like local optimization with adaptive learning rate.
    """
    r = U.shape[2]
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Adam optimizer state
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    for step in range(steps):
        if _time_left() < 5:
            break
        
        t += 1
        
        # Compute T and error
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        diff = T - 1.0
        err = np.sum(diff * diff)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-12:
            break
        
        # Gradients
        dU = np.einsum('ijk,jkc,kic->ijc', diff, V, W)
        dV = np.einsum('ijk,ijc,kic->jkc', diff, U, W)
        dW = np.einsum('ijk,ijc,jkc->kic', diff, U, V)
        
        # Adam update
        mU = beta1 * mU + (1 - beta1) * dU
        vU = beta2 * vU + (1 - beta2) * dU**2
        mU_hat = mU / (1 - beta1**t)
        vU_hat = vU / (1 - beta2**t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = beta1 * mV + (1 - beta1) * dV
        vV = beta2 * vV + (1 - beta2) * dV**2
        mV_hat = mV / (1 - beta1**t)
        vV_hat = vV / (1 - beta2**t)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = beta1 * mW + (1 - beta1) * dW
        vW = beta2 * vW + (1 - beta2) * dW**2
        mW_hat = mW / (1 - beta1**t)
        vW_hat = vW / (1 - beta2**t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        # Decay learning rate
        lr *= 0.995
        
    return best_U, best_V, best_W, best_err

def _try_kronecker_structure(n, m, p, r, seed=42):
    """
    Try Kronecker product structure: exploit that 4x5x5 can be viewed
    as combinations of smaller matrix multiplications.
    """
    np.random.seed(seed)
    
    # Decompose dimensions: 4 = 2*2, 5 = 5, 5 = 5
    # Try to use rank-1 building blocks with shared structure
    
    U = np.zeros((n, m, r))
    V = np.zeros((m, p, r))
    W = np.zeros((p, n, r))
    
    # Assign each rank component a "role"
    # For c in range(r):
    #   U[:, :, c] = outer(a_c, b_c) where a_c in R^n, b_c in R^m
    #   V[:, :, c] = outer(c_c, d_c) where c_c in R^m, d_c in R^p
    #   W[:, :, c] = outer(e_c, f_c) where e_c in R^p, f_c in R^n
    
    # Initialize with random vectors
    for c in range(r):
        a_c = np.random.randn(n) * 0.5
        b_c = np.random.randn(m) * 0.5
        c_c = np.random.randn(m) * 0.5
        d_c = np.random.randn(p) * 0.5
        e_c = np.random.randn(p) * 0.5
        f_c = np.random.randn(n) * 0.5
        
        U[:, :, c] = np.outer(a_c, b_c)
        V[:, :, c] = np.outer(c_c, d_c)
        W[:, :, c] = np.outer(e_c, f_c)
    
    # Optimize
    best_U, best_V, best_W, best_err = _local_optimize(U.copy(), V.copy(), W.copy(), n, m, p, steps=300)
    
    return best_U, best_V, best_W, best_err

def _try_direct_search(n, m, p, r, seed=42, init_scale=0.1):
    """
    Direct random initialization with Adam optimization.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n, m, r) * init_scale
    V = np.random.randn(m, p, r) * init_scale
    W = np.random.randn(p, n, r) * init_scale
    
    best_U, best_V, best_W, best_err = _local_optimize(U.copy(), V.copy(), W.copy(), n, m, p, steps=400)
    
    return best_U, best_V, best_W, best_err

def _polish(U, V, W, n, m, p, steps=100, lr=0.001):
    """
    High-precision polish pass with very small learning rate.
    """
    r = U.shape[2]
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(steps):
        if _time_left() < 3:
            break
        
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        diff = T - 1.0
        err = np.sum(diff * diff)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-14:
            break
        
        dU = np.einsum('ijk,jkc,kic->ijc', diff, V, W)
        dV = np.einsum('ijk,ijc,kic->jkc', diff, U, W)
        dW = np.einsum('ijk,ijc,jkc->kic', diff, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive lr decay
        lr *= 0.99
        
    return best_U, best_V, best_W, best_err

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured approaches first (faster convergence)
    # Then direct search for lower ranks
    
    # Rank candidates to try: start from plausible lower bounds
    # For 4x5x5, known lower bounds suggest rank around 80-100
    # Try structured approaches at various ranks
    
    ranks_to_try = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 88, 85, 80]
    
    for r in ranks_to_try:
        if _time_left() < 15:
            break
        
        if r >= best_rank:
            continue
            
        # Try multiple strategies for this rank
        strategies = [
            ("structured", lambda s: _try_structured_winograd(n, m, p, r, seed=s)),
            ("kronecker", lambda s: _try_kronecker_structure(n, m, p, r, seed=s)),
            ("direct", lambda s: _try_direct_search(n, m, p, r, seed=s)),
        ]
        
        found = False
        for name, strategy_fn in strategies:
            if _time_left() < 8:
                break
                
            for seed in range(3):
                if _time_left() < 5:
                    break
                    
                U, V, W, err = strategy_fn(seed)
                
                if err < 1e-8:
                    # Polish for higher precision
                    U, V, W, err = _polish(U, V, W, n, m, p, steps=50, lr=0.0005)
                    
                    if err < 1e-10:
                        U_list = U.reshape(n*m, r).tolist()
                        V_list = V.reshape(m*p, r).tolist()
                        W_list = W.reshape(p*n, r).tolist()
                        
                        if _verify(U_list, V_list, W_list, n, m, p):
                            best = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = r
                            found = True
                            break
                            
            if found:
                break
    
    return best
# EVOLVE-BLOCK-END
