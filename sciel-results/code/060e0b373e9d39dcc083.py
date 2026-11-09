"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] using einsum."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    return np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)


def _verify_decomposition(decomp, T, tol=1e-9):
    """Check if the decomposition exactly reproduces T."""
    U, V, W = decomp['U'], decomp['V'], decomp['W']
    T_approx = _tensor_from_decomp(U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)


def _loss_and_grad(U, V, W, T):
    """
    Compute loss ||T - sum_r U_r V_r W_r||^2 and gradients dU, dV, dW.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    # Compute predicted tensor
    T_pred = np.einsum('ar,br,cr->abc', U, V, W)
    diff = T - T_pred
    
    # Loss
    loss = np.sum(diff ** 2)
    
    # Gradients
    # dU[a, r] = -2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    # dV[b, r] = -2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
    # dW[c, r] = -2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
    
    dU = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
    dV = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
    dW = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    return loss, dU, dV, dW


def _optimize_decomposition(n, m, p, T, R, max_iters=1000, lr=0.01, tol=1e-12, deadline=None):
    """
    Optimize decomposition of rank R using gradient descent.
    """
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        loss, dU, dV, dW = _loss_and_grad(U, V, W, T)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_loss < tol:
            break
            
        # Gradient descent step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    return best_U, best_V, best_W, best_loss


def _snap_to_half_integers(U, V, W, tol=1e-6):
    """
    Snap values to nearest half-integer if close enough.
    """
    def snap(arr):
        # Multiply by 2, round to nearest integer, divide by 2
        arr_2 = arr * 2
        arr_rounded = np.round(arr_2)
        # Check if close to half-integer
        if np.allclose(arr_2, arr_rounded, atol=tol):
            return arr_rounded / 2.0
        else:
            return arr
    
    U_snapped = snap(U)
    V_snapped = snap(V)
    W_snapped = snap(W)
    
    # Verify the snapped version still works
    T_pred = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
    
    return U_snapped, V_snapped, W_snapped, T_pred


def _polish_decomposition(n, m, p, T, U, V, W, max_iters=100, tol=1e-14, deadline=None):
    """
    High-precision polish pass using Newton-like method or tight gradient descent.
    """
    # For now, just do more gradient descent with smaller step size
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        loss, dU, dV, dW = _loss_and_grad(U, V, W, T)
        
        if loss < tol:
            break
            
        # Adaptive step size
        lr = 0.001 * (1.0 + loss)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    return U, V, W, loss


def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Search for a decomposition with rank < n*m*p.
    """
    schoolbook_rank = n * m * p
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from schoolbook_rank - 1 down to 1
    # But we start with a reasonable guess. For 3x4x8, schoolbook is 96.
    # Known bounds might be lower, but we don't know them.
    # Let's try a few ranks.
    
    best_rank = schoolbook_rank
    best_decomp = _schoolbook(n, m, p)
    
    # We need to be careful with time.
    # Let's try ranks: 95, 94, ... but that's too many.
    # Maybe try a few specific ranks or a binary search?
    # Since we don't know the structure, random search at lower ranks might not work well.
    
    # Let's just try to improve the schoolbook rank by 1 or 2 if possible?
    # Actually, the problem asks for rank-minimal.
    # If we can't find a better one, we return schoolbook.
    
    # Let's try rank = schoolbook_rank - 1
    # And maybe a few others.
    
    ranks_to_try = [schoolbook_rank - 1, schoolbook_rank - 2, schoolbook_rank - 5, schoolbook_rank - 10]
    
    for R in ranks_to_try:
        if R < 1:
            continue
            
        if time.time() > deadline:
            break
            
        # Try to optimize
        U, V, W, loss = _optimize_decomposition(n, m, p, T, R, max_iters=200, lr=0.01, tol=1e-10, deadline=deadline)
        
        if loss < 1e-8:
            # Try to snap to half-integers
            U_s, V_s, W_s, T_pred = _snap_to_half_integers(U, V, W)
            if np.allclose(T_pred, T, atol=1e-9, rtol=0):
                # Success!
                # Convert to lists of Fractions or floats?
                # The task says "integer/half-integer entries".
                # Returning floats that are exactly half-integers should be fine if they are exact.
                # But to be safe, let's return the snapped values.
                decomp = {
                    "U": U_s.tolist(),
                    "V": V_s.tolist(),
                    "W": W_s.tolist()
                }
                return R, decomp
        
        # If optimization didn't reach low loss, try polishing?
        # No, if it didn't reach low loss, it's not a valid decomposition.
        
    # If no better rank found, return schoolbook
    return schoolbook_rank, best_decomp


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 40s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    # Verify the decomposition to ensure it is correct
    T = _build_target_tensor(n, m, p)
    if not _verify_decomposition(decomp, T):
        # Fallback to schoolbook if verification fails (should not happen)
        decomp = _schoolbook(n, m, p)
    
    return decomp
# EVOLVE-BLOCK-END
