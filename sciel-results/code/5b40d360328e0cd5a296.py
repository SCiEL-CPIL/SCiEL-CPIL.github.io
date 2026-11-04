"""Rank-minimal tensor decomposition for 3x3x8 matrix multiplication.

Uses numerical optimization to find low-rank decompositions, then rationalizes
to exact integer/half-integer entries.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [3, 3, 8]
DEADLINE = 140.0  # Safety margin before 150s kill

def _target_tensor():
    """The multiplication tensor: T[i,j,k] = 1 for all valid i,j,k."""
    n, m, p = DIMENSIONS
    return np.ones((n, m, p))

def _compute_tensor(U, V, W):
    """Compute the tensor from factor matrices using trace convention."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    return tensor

def _residual(U, V, W):
    """Compute max residual between computed tensor and target."""
    n, m, p = DIMENSIONS
    target = np.ones((n, m, p))
    computed = _compute_tensor(U, V, W)
    return np.max(np.abs(computed - target))

def _rationalize_matrix(M, tol=1e-8):
    """
    Attempt to rationalize a real matrix to integer or half-integer entries.
    Returns (rational_matrix, success) where rational_matrix has Fraction entries.
    """
    n, m = M.shape
    rational = np.zeros((n, m), dtype=object)
    
    for i in range(n):
        for j in range(m):
            val = M[i, j]
            if abs(val) < tol:
                rational[i, j] = Fraction(0, 1)
            else:
                # Try to find a fraction with small denominator
                # First try integer
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    rational[i, j] = Fraction(int(nearest_int), 1)
                else:
                    # Try half-integer
                    nearest_half = round(2 * val) / 2
                    if abs(val - nearest_half) < tol:
                        rational[i, j] = Fraction(int(round(2 * val)), 2)
                    else:
                        # Try continued fraction with small denominator
                        found = False
                        for denom in range(1, 1000):
                            numer = round(val * denom)
                            if abs(val - numer / denom) < tol:
                                rational[i, j] = Fraction(numer, denom)
                                found = True
                                break
                        if not found:
                            # Fallback: use Fraction directly
                            rational[i, j] = Fraction(val).limit_denominator(10000)
    
    return rational

def _verify_exact(U_frac, V_frac, W_frac):
    """
    Verify that the rational factor matrices exactly produce the target tensor.
    U_frac, V_frac, W_frac are 2D arrays of Fraction objects.
    """
    n, m, p = DIMENSIONS
    rank = U_frac.shape[1]
    target = np.ones((n, m, p), dtype=object)
    
    computed = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed[i, j, k] += (U_frac[i * m + j, r] * 
                                          V_frac[j * p + k, r] * 
                                          W_frac[k * n + i, r])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if computed[i, j, k] != target[i, j, k]:
                    return False
    return True

def _build_schoolbook():
    """Build the schoolbook decomposition (rank = n*m*p)."""
    n, m, p = DIMENSIONS
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

def _optimize_rank(U_init, V_init, W_init, max_iter=500, lr=0.01, tol=1e-10):
    """
    Simple gradient descent to minimize the residual.
    Returns optimized U, V, W and final residual.
    """
    n, m, p = DIMENSIONS
    rank = U_init.shape[1]
    
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    
    best_residual = _residual(U, V, W)
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Compute current tensor
        current = _compute_tensor(U, V, W)
        target = np.ones((n, m, p))
        diff = current - target
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dU[i * m + j, r] += diff[i, j, k] * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += diff[i, j, k] * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += diff[i, j, k] * U[i * m + j, r] * V[j * p + k, r]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        current_residual = _residual(U, V, W)
        if current_residual < best_residual:
            best_residual = current_residual
        
        if best_residual < tol:
            break
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.5
    
    return U, V, W, best_residual

def _try_rank_decomposition(target_rank, max_attempts=20):
    """
    Try to find a decomposition of the given rank.
    Returns (U, V, W, success) where U,V,W are numpy arrays or None.
    """
    n, m, p = DIMENSIONS
    np.random.seed(42)
    
    best_U, best_V, best_W = None, None, None
    best_residual = np.inf
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Optimize
        U_opt, V_opt, W_opt, residual = _optimize_rank(U, V, W, max_iter=200, lr=0.05)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U_opt, V_opt, W_opt
        
        if best_residual < 1e-10:
            break
    
    if best_residual < 1e-10:
        return best_U, best_V, best_W, True
    return None, None, None, False

def _build_rational_from_real(U_real, V_real, W_real):
    """
    Convert real-valued factors to rational (Fraction) factors.
    Returns (U_frac, V_frac, W_frac, success).
    """
    U_frac = _rationalize_matrix(U_real)
    V_frac = _rationalize_matrix(V_real)
    W_frac = _rationalize_matrix(W_real)
    
    success = _verify_exact(U_frac, V_frac, W_frac)
    return U_frac, V_frac, W_frac, success

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook()
    school_rank = U_school.shape[1]
    
    best_rank = school_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Start from rank-1 and go up, but for efficiency, try a few key ranks
    ranks_to_try = list(range(1, min(school_rank, 30)))
    
    for target_rank in ranks_to_try:
        if time.time() - t0 > DEADLINE - 5:
            break
        
        U, V, W, success = _try_rank_decomposition(target_rank, max_attempts=10)
        
        if success and U is not None:
            # Try to rationalize
            U_frac, V_frac, W_frac, rational_success = _build_rational_from_real(U, V, W)
            
            if rational_success:
                # Convert to lists of floats (integers or half-integers)
                best_rank = target_rank
                best_U = np.array([[float(x) for x in row] for row in U_frac], dtype=float)
                best_V = np.array([[float(x) for x in row] for row in V_frac], dtype=float)
                best_W = np.array([[float(x) for x in row] for row in W_frac], dtype=float)
                break  # Found a valid lower rank, stop
    
    # If no lower rank found, use schoolbook
    if best_rank == school_rank:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    if time.time() - t0 > DEADLINE:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
