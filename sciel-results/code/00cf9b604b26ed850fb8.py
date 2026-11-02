"""Tensor decomposition search for 3x3 * 3x8 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-72 decomposition."""
    n, m, p = 3, 3, 8
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_optimize(U, V, W, n, m, p, max_iters=1000, tol=1e-10, deadline=None):
    """Alternating least squares optimization for tensor decomposition.
    
    For the all-ones tensor T[i,j,k]=1, we need:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    This is a 3-way tensor completion problem. We use ALS by fixing two factors
    and solving for the third.
    """
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    rank = Ua.shape[1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current predictions
        # pred[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        pred = np.zeros((n, m, p))
        for c in range(rank):
            # Outer product of U[:,c], V[:,c], W[:,c] with proper indexing
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        pred[i, j, k] += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
        
        err = np.sum((pred - 1.0) ** 2)
        if err < tol:
            break
        
        # Update W: fix U and V, solve for W
        # For each (i,k) pair in W, we have:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
        # This gives m equations for each (i,k)
        # But W[k*n+i,c] appears in all j equations for that (i,k)
        
        # Actually, let's think differently. For fixed U,V, the constraint is:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by (i,k): for fixed i,k, we have m equations (one per j)
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0,...,m-1
        
        # This is underdetermined in general, but we can use least squares
        # Or better: use the fact that we want to minimize sum over all (i,j,k)
        
        # Let's use a different approach: gradient-based with line search
        lr = 0.01 / (1 + it * 0.05)
        
        # Compute gradients
        # dE/dW[k*n+i,c] = 2 * sum_j (pred[i,j,k] - 1) * U[i*m+j,c] * V[j*p+k,c]
        grad_W = np.zeros_like(Wa)
        grad_U = np.zeros_like(Ua)
        grad_V = np.zeros_like(Va)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = pred[i, j, k] - 1.0
                    for c in range(rank):
                        grad_W[k * n + i, c] += 2 * diff * Ua[i * m + j, c] * Va[j * p + k, c]
                        grad_U[i * m + j, c] += 2 * diff * Va[j * p + k, c] * Wa[k * n + i, c]
                        grad_V[j * p + k, c] += 2 * diff * Ua[i * m + j, c] * Wa[k * n + i, c]
        
        # Gradient descent step
        Ua -= lr * grad_U
        Va -= lr * grad_V
        Wa -= lr * grad_W
    
    return Ua, Va, Wa

def _find_decomposition_rank(r, n, m, p, max_restarts=5, max_iters=500, deadline=None):
    """Try to find a rank-r decomposition using random initialization + ALS."""
    rng = np.random.RandomState(42)
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # Optimize
        Ua, Va, Wa = _als_optimize(U, V, W, n, m, p, max_iters=max_iters, tol=1e-10, deadline=deadline)
        
        # Check if we found an exact solution
        if _verify_decomposition(Ua.tolist(), Va.tolist(), Wa.tolist(), n, m, p, tol=1e-8):
            return {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit (5s safety margin)
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook as safe baseline
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _find_decomposition_rank(r, n, m, p, max_restarts=3, max_iters=200, deadline=deadline)
        if result is not None:
            # Verify and return
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                return result
    
    return best
# EVOLVE-BLOCK-END
