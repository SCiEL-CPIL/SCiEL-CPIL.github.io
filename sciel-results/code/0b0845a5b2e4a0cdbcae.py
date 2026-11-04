"""Matrix multiplication 2x5x5 rank minimization via iterative truncated SVD and structured search."""
DIMENSIONS = [2, 5, 5]

import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook_decomposition():
    """Return the standard schoolbook decomposition of rank 50."""
    n, m, p = DIMENSIONS
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
    return U, V, W


def _build_tensor():
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    where T[i*m+j, j*p+k, k*n+i] = 1.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-9):
    """
    Verify that U, V, W exactly decompose tensor T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    # Compute reconstruction
    # Efficiently: for each r, outer product U[:,r] * V[:,r] * W[:,r]
    recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
    return np.allclose(recon, T, atol=tol)


def _try_rank_truncated_svd(T, target_rank, max_iter=200, tol=1e-8, deadline=None):
    """
    Try to find a decomposition of rank target_rank using iterative truncated SVD.
    Uses the approach of alternating least squares / truncated SVD on matricizations.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize randomly
    U = np.random.randn(nm, target_rank)
    V = np.random.randn(mp, target_rank)
    W = np.random.randn(pn, target_rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each column r, U[:,r] should approximate sum over b,c of T[:,b,c]*V[b,r]*W[c,r]
        # Matricization: T_1[b,c] = T[:,b,c].sum over a? No.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Fix V,W: U[:,r] = argmin ||T - sum_r' U[:,r'] V[:,r'] W[:,r']||
        # For each r: U[:,r] = (V[:,r] W[:,r] matricized)^-1 * T contracted
        # More precisely: let M_r = outer(V[:,r], W[:,r]) reshaped to (nm*mp*pn)
        # This is expensive. Use iterative approach instead.
        
        # Alternate: update U given V,W
        # T = U @ (V * W)^T in tensor sense
        # For each r: U[:,r] = pinv(V[:,r] * W[:,r] as linear op) * T
        # Simpler: use gradient-free alternating minimization
        
        # Update U:
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed V,W: U[:,r] = argmin_a sum_{b,c} (T[a,b,c] - sum_r' U[a,r']V[b,r']W[c,r'])^2
        # This gives for each r: U[:,r] = (sum_{b,c} V[b,r]W[c,r] V[b,r']W[c,r'] * T[a,b,c]) / (sum_{b,c} V[b,r]^2 W[c,r]^2)
        
        for r in range(target_rank):
            if deadline is not None and time.time() > deadline:
                break
            # Compute numerator: for each a, sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
            num = np.einsum('abc,b,c->a', T, V[:, r], W[:, r])
            # Denominator: sum_{b,c} V[b,r]^2 * W[c,r]^2
            den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-12:
                U[:, r] = num / den
        
        # Update V:
        for r in range(target_rank):
            if deadline is not None and time.time() > deadline:
                break
            num = np.einsum('abc,a,c->b', T, U[:, r], W[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-12:
                V[:, r] = num / den
        
        # Update W:
        for r in range(target_rank):
            if deadline is not None and time.time() > deadline:
                break
            num = np.einsum('abc,a,b->c', T, U[:, r], V[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
            if den > 1e-12:
                W[:, r] = num / den
        
        # Compute error
        recon = np.zeros_like(T)
        for r in range(target_rank):
            recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    if best_err < tol:
        return best_U, best_V, best_W, True
    return best_U, best_V, best_W, False


def _try_rank_random_search(T, target_rank, n_trials=10, max_iter=100, tol=1e-8, deadline=None):
    """Try random initialization with alternating minimization."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            for r in range(target_rank):
                num = np.einsum('abc,b,c->a', T, V[:, r], W[:, r])
                den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
                if den > 1e-12:
                    U[:, r] = num / den
            
            for r in range(target_rank):
                num = np.einsum('abc,a,c->b', T, U[:, r], W[:, r])
                den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
                if den > 1e-12:
                    V[:, r] = num / den
            
            for r in range(target_rank):
                num = np.einsum('abc,a,b->c', T, U[:, r], V[:, r])
                den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
                if den > 1e-12:
                    W[:, r] = num / den
            
            recon = np.zeros_like(T)
            for r in range(target_rank):
                recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                break
        
        if best_err < tol:
            return best_U, best_V, best_W, True
    
    return None, None, None, False


def _to_half_integer_matrix(M):
    """Convert a float matrix to half-integer entries (multiply by 2, round, divide by 2)."""
    M2 = np.round(M * 2) / 2
    return M2


def _verify_half_integer(U, V, W, T, tol=1e-9):
    """Verify decomposition with half-integer constraint."""
    recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
    return np.allclose(recon, T, atol=tol)


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with shapes (10, rank), (25, rank), (10, rank).
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the tensor
    T = _build_tensor()
    
    # Start with schoolbook decomposition
    U0, V0, W0 = _schoolbook_decomposition()
    best_rank = U0.shape[1]  # 50
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 2x5x5, the rank is at least 25 (n*m*p / max(n,m,p))
    # Actually, the trivial lower bound is max(nm, mp, pn) = 25
    # Try ranks from 25 upward
    
    for target_rank in range(25, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try truncated SVD approach
        success = False
        try:
            U, V, W, success = _try_rank_truncated_svd(T, target_rank, max_iter=100, tol=1e-8, deadline=deadline)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        except Exception:
            pass
        
        # Try random search
        if time.time() > deadline - 10:
            break
        try:
            U, V, W, success = _try_rank_random_search(T, target_rank, n_trials=5, max_iter=80, tol=1e-8, deadline=deadline)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        except Exception:
            pass
    
    # Convert to half-integer if possible
    # Check if the solution can be represented with half-integers
    # For now, return as-is since the problem allows integer/half-integer entries
    # The schoolbook solution uses integers, which are valid half-integers
    
    # Return as lists for compatibility
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
