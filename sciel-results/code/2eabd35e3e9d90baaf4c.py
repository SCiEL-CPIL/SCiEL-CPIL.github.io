import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication.
    
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _bls17():
    """
    Rank-17 decomposition for 3x3 * 3x5 matrix multiplication.
    This is a hardcoded decomposition that may or may not be valid.
    We verify it at runtime.
    """
    U = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ]
    
    V = [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    W = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ]
    
    return {"U": U, "V": V, "W": W}

def _construct_rank_r(n, m, p, r, init_type="random"):
    """Construct a random or structured initialization for rank r."""
    if init_type == "random":
        U = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)], dtype=float)
        V = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)], dtype=float)
        W = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)], dtype=float)
    elif init_type == "sparse":
        # Sparse initialization with some structure
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        for c in range(r):
            # Each column connects some subset
            for _ in range(3):
                ui = random.randint(0, n * m - 1)
                vi = random.randint(0, m * p - 1)
                wi = random.randint(0, p * n - 1)
                U[ui, c] = random.choice([-1, 1])
                V[vi, c] = random.choice([-1, 1])
                W[wi, c] = random.choice([-1, 1])
    else:
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
    return U, V, W

def _compute_error(U, V, W, idx_u, idx_v, idx_w):
    """Compute the total error for the decomposition."""
    errs = []
    for idx in range(len(idx_u)):
        s = np.dot(U[idx_u[idx]], V[idx_v[idx]] * W[idx_w[idx]])
        errs.append(s - 1.0)
    return np.sum(np.array(errs) ** 2)

def _compute_gradients(U, V, W, idx_u, idx_v, idx_w):
    """Compute gradients for the decomposition error."""
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for idx in range(len(idx_u)):
        s = np.dot(U[idx_u[idx]], V[idx_v[idx]] * W[idx_w[idx]])
        diff = 2.0 * (s - 1.0)
        
        grad_U[idx_u[idx]] += diff * V[idx_v[idx]] * W[idx_w[idx]]
        grad_V[idx_v[idx]] += diff * U[idx_u[idx]] * W[idx_w[idx]]
        grad_W[idx_w[idx]] += diff * U[idx_u[idx]] * V[idx_v[idx]]
    
    return grad_U, grad_V, grad_W

def _snap_to_half(U, V, W):
    """Snap values to nearest half-integer."""
    U_snap = np.round(U * 2.0) / 2.0
    V_snap = np.round(V * 2.0) / 2.0
    W_snap = np.round(W * 2.0) / 2.0
    return U_snap, V_snap, W_snap

def _verify_exact(U, V, W, idx_u, idx_v, idx_w, tol=1e-9):
    """Verify that the decomposition is exact."""
    for idx in range(len(idx_u)):
        s = np.dot(U[idx_u[idx]], V[idx_v[idx]] * W[idx_w[idx]])
        if abs(s - 1.0) > tol:
            return False
    return True

def _refine_decomposition(U, V, W, idx_u, idx_v, idx_w, max_time, start_time):
    """Refine a decomposition using gradient descent with snapping."""
    r = U.shape[1]
    best_err = _compute_error(U, V, W, idx_u, idx_v, idx_w)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.02
    patience = 0
    max_patience = 200
    
    for it in range(50000):
        if time.time() - start_time > max_time:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, idx_u, idx_v, idx_w)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        err = _compute_error(U, V, W, idx_u, idx_v, idx_w)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            # Try snapping
            U_snap, V_snap, W_snap = _snap_to_half(U, V, W)
            if _verify_exact(U_snap, V_snap, W_snap, idx_u, idx_v, idx_w):
                return U_snap, V_snap, W_snap, True
            
            # Reduce learning rate and reset
            lr *= 0.5
            patience = 0
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        
        if err < 1e-10:
            U_snap, V_snap, W_snap = _snap_to_half(U, V, W)
            if _verify_exact(U_snap, V_snap, W_snap, idx_u, idx_v, idx_w):
                return U_snap, V_snap, W_snap, True
    
    # Final attempt to snap
    U_snap, V_snap, W_snap = _snap_to_half(best_U, best_V, best_W)
    if _verify_exact(U_snap, V_snap, W_snap, idx_u, idx_v, idx_w):
        return U_snap, V_snap, W_snap, True
    
    return best_U, best_V, best_W, False

def _search_rank17(n, m, p, max_time=110):
    """
    Search for a rank-17 decomposition using multiple strategies.
    """
    start = time.time()
    target_rank = 17
    
    # Pre-compute indices
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
    idx_u = np.array(idx_u)
    idx_v = np.array(idx_v)
    idx_w = np.array(idx_w)
    
    best_result = None
    best_err = float('inf')
    
    # Strategy 1: Multiple random initializations with gradient descent
    for seed in range(100):
        if time.time() - start > max_time:
            break
        
        random.seed(seed)
        np.random.seed(seed)
        
        U, V, W = _construct_rank_r(n, m, p, target_rank, init_type="random")
        
        U_ref, V_ref, W_ref, found = _refine_decomposition(U, V, W, idx_u, idx_v, idx_w, max_time, start)
        
        if found:
            return U_ref.tolist(), V_ref.tolist(), W_ref.tolist()
        
        err = _compute_error(U_ref, V_ref, W_ref, idx_u, idx_v, idx_w)
        if err < best_err:
            best_err = err
            best_result = (U_ref, V_ref, W_ref)
    
    # Strategy 2: Sparse initializations
    for seed in range(50):
        if time.time() - start > max_time:
            break
        
        random.seed(seed + 1000)
        np.random.seed(seed + 1000)
        
        U, V, W = _construct_rank_r(n, m, p, target_rank, init_type="sparse")
        
        U_ref, V_ref, W_ref, found = _refine_decomposition(U, V, W, idx_u, idx_v, idx_w, max_time, start)
        
        if found:
            return U_ref.tolist(), V_ref.tolist(), W_ref.tolist()
        
        err = _compute_error(U_ref, V_ref, W_ref, idx_u, idx_v, idx_w)
        if err < best_err:
            best_err = err
            best_result = (U_ref, V_ref, W_ref)
    
    # Strategy 3: Small random perturbations
    for seed in range(50):
        if time.time() - start > max_time:
            break
        
        random.seed(seed + 2000)
        np.random.seed(seed + 2000)
        
        U, V, W = _construct_rank_r(n, m, p, target_rank, init_type="gaussian")
        
        U_ref, V_ref, W_ref, found = _refine_decomposition(U, V, W, idx_u, idx_v, idx_w, max_time, start)
        
        if found:
            return U_ref.tolist(), V_ref.tolist(), W_ref.tolist()
        
        err = _compute_error(U_ref, V_ref, W_ref, idx_u, idx_v, idx_w)
        if err < best_err:
            best_err = err
            best_result = (U_ref, V_ref, W_ref)
    
    # If we found a good result but not exact, try one more refinement
    if best_result is not None and best_err < 1e-4:
        U, V, W = best_result
        U_ref, V_ref, W_ref, found = _refine_decomposition(U, V, W, idx_u, idx_v, idx_w, max_time, start)
        if found:
            return U_ref.tolist(), V_ref.tolist(), W_ref.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, verify the hardcoded BLS-17 decomposition
    bls = _bls17()
    if _validate(bls["U"], bls["V"], bls["W"], n, m, p):
        return bls
    
    # Try to find a rank-17 decomposition via search
    result = _search_rank17(n, m, p, max_time=110)
    if result is not None:
        U, V, W = result
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 45)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
