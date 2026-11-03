import time
import numpy as np
import scipy.optimize as optimize
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
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
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor as a 3D array.
    T[i,j,k] = 1 if (i,j,k) is a valid multiplication term, else 0.
    Actually, for matmul, the tensor has T[i,j,k]=1 for all valid (i,j,k).
    """
    T = np.ones((n, m, p))
    return T

def _tensor_to_equations(U, V, W, n, m, p):
    """
    Given U (n*m, rank), V (m*p, rank), W (p*n, rank),
    check if they decompose the matmul tensor.
    Returns the residual vector of equation violations.
    """
    rank = U.shape[1]
    residuals = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                residuals.append(s - 1.0)
    
    return np.array(residuals)

def _try_rank_r(r_target, n=2, m=3, p=6, max_iters=50, time_budget=30.0):
    """
    Try to find a rank-r decomposition using gradient descent.
    Returns (U, V, W) or None.
    """
    if time.time() - START_TIME > TIME_LIMIT - time_budget:
        return None
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, r_target) * 0.1
    V = np.random.randn(m * p, r_target) * 0.1
    W = np.random.randn(p * n, r_target) * 0.1
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Compute residuals
        residuals = _tensor_to_equations(U, V, W, n, m, p)
        current_residual = np.sum(residuals ** 2)
        
        if current_residual < best_residual:
            best_residual = current_residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if current_residual < 1e-10:
            break
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    residual = residuals[i * m * p + j * p + k]
                    
                    # d/dU[idx, r] = residual * V[v_idx, r] * W[w_idx, r]
                    grad_U[idx, :] += residual * V[v_idx, :] * W[w_idx, :]
                    grad_V[v_idx, :] += residual * U[idx, :] * W[w_idx, :]
                    grad_W[w_idx, :] += residual * U[idx, :] * V[v_idx, :]
        
        # Gradient descent with adaptive step size
        lr = 0.01 / (1 + it * 0.1)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    # Check if we found a valid solution
    if best_residual < 1e-8:
        # Snap to half-integers
        U_snapped = _snap_to_half_integers(best_U)
        V_snapped = _snap_to_half_integers(best_V)
        W_snapped = _snap_to_half_integers(best_W)
        
        if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p):
            return U_snapped, V_snapped, W_snapped
    
    return None

def _try_structural_rank(r_target, n=2, m=3, p=6, time_budget=25.0):
    """
    Try structured decompositions that exploit the matrix multiplication structure.
    For 2x3x6, try decompositions with specific patterns.
    """
    if time.time() - START_TIME > TIME_LIMIT - time_budget:
        return None
    
    # Try a block structure: decompose the 6 columns into two groups of 3
    # This might allow rank reduction if the two 2x3x3 blocks share structure
    
    # Strategy: Assume the decomposition can be written as a sum of terms
    # where each term has a specific sparsity pattern
    
    # For now, try a simple approach: use the fact that 6 = 2*3
    # and see if we can find a rank-34 or lower solution
    
    # Initialize with a structured pattern
    U = np.zeros((n * m, r_target))
    V = np.zeros((m * p, r_target))
    W = np.zeros((p * n, r_target))
    
    # Fill with a pattern based on modular arithmetic
    for r in range(r_target):
        i = r % n
        j = (r // n) % m
        k1 = (r // (n * m)) % 3
        k2 = (r // (n * m * 3)) % 2
        
        U[i * m + j, r] = 1.0
        V[j * p + k1, r] = 1.0
        W[(k1 + 3 * k2) * n + i, r] = 1.0
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 2x3x6 is not well-known,
    # but we can try ranks from 34 down to 30
    
    for r in range(34, 29, -1):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        # Try structural approach first (faster)
        result = _try_structural_rank(r, n, m, p, time_budget=5.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        # Try gradient descent approach
        result = _try_rank_r(r, n, m, p, max_iters=30, time_budget=10.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
