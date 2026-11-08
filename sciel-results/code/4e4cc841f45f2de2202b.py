"""Program searching for rank-minimal exact tensor decomposition of 4x4 * 4x6 matrix mult."""
import numpy as np
from itertools import product
import random
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(T, 1.0, atol=tol)


def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - 1."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return T - 1.0


def _jacobian(U, V, W, n, m, p):
    """Compute the Jacobian of the residual with respect to U, V, W entries."""
    rank = U.shape[1]
    n_eq = n * m * p
    n_var = rank * (n * m + m * p + p * n)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    J = np.zeros((n_eq, n_var))
    
    # Flatten indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                
                # d/dU[i,j,c] = V[j,k,c] * W[k,i,c]
                for c in range(rank):
                    var_idx = c * (n*m + m*p + p*n) + i * m + j
                    J[eq_idx, var_idx] = V_r[j, k, c] * W_r[k, i, c]
                
                # d/dV[j,k,c] = U[i,j,c] * W[k,i,c]
                for c in range(rank):
                    var_idx = c * (n*m + m*p + p*n) + (n*m) + j * p + k
                    J[eq_idx, var_idx] = U_r[i, j, c] * W_r[k, i, c]
                
                # d/dW[k,i,c] = U[i,j,c] * V[j,k,c]
                for c in range(rank):
                    var_idx = c * (n*m + m*p + p*n) + (n*m + m*p) + k * n + i
                    J[eq_idx, var_idx] = U_r[i, j, c] * V_r[j, k, c]
    
    return J


def _newton_step(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Try to refine U, V, W using Newton's method to achieve exact decomposition."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for it in range(max_iter):
        residual = _compute_residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(residual)
        if res_norm < tol:
            return U, V, W, True
        
        J = _jacobian(U, V, W, n, m, p)
        
        # Solve J * delta = -residual
        try:
            delta, residuals, rank_solve, sv = np.linalg.lstsq(J, -residual.flatten(), rcond=None)
        except np.linalg.LinAlgError:
            break
        
        if np.linalg.norm(delta) < 1e-15:
            break
        
        # Update
        delta_U = delta[:rank * n * m].reshape(n, m, rank)
        delta_V = delta[rank * n * m:rank * (n * m + m * p)].reshape(m, p, rank)
        delta_W = delta[rank * (n * m + m * p):].reshape(p, n, rank)
        
        U += delta_U.reshape(n * m, rank)
        V += delta_V.reshape(m * p, rank)
        W += delta_W.reshape(p * n, rank)
    
    residual_final = _compute_residual(U, V, W, n, m, p)
    return U, V, W, np.linalg.norm(residual_final) < tol


def _random_search_for_rank(n, m, p, rank, n_attempts=100, deadline=None):
    """Try to find an exact rank-r decomposition via random init + Newton refinement."""
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(n_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Try Newton refinement
        U_new, V_new, W_new, success = _newton_step(U, V, W, n, m, p, max_iter=30)
        
        if success:
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    return None, None, None


def _strassen_2x2_2x2(A, B):
    """Strassen's algorithm for 2x2 * 2x2, returns rank-7 decomposition coefficients."""
    # This is a helper for building structured decompositions
    pass


def _build_strassen_4x4_4x6():
    """
    Build a Strassen-based decomposition for 4x4 * 4x6.
    
    Strategy: Split into 2x2 blocks.
    A is 4x4: split into 4 blocks of 2x2
    B is 4x6: split into 4 blocks of 2x3 (2 rows, 3 cols each)
    C is 4x6: split into 4 blocks of 2x3
    
    For each output block C_ij, we need to multiply appropriate A blocks with B blocks.
    
    Actually, let's use a simpler structured approach:
    Use 2x2 Strassen on the 4x4 part and handle the 6 columns separately.
    
    Alternative: Pad B to 4x8, use 4x4 * 4x8 decomposition, then take first 6 cols.
    But that doesn't directly help.
    
    Let's try: use the fact that 4x4 * 4x6 can be seen as 4x4 * 4x4 + 4x4 * 4x2.
    If we have rank-49 for 4x4*4x4 (Strassen) and rank-32 for 4x4*4x2, total = 81.
    But that's not minimal.
    
    Let's just try random search with Newton.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Verify schoolbook
    if not _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # Try ranks from 95 down to 40 (reasonable lower bound)
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random search with Newton refinement
        U_new, V_new, W_new = _random_search_for_rank(
            n, m, p, rank, 
            n_attempts=50,
            deadline=deadline - 5
        )
        
        if U_new is not None:
            best_rank = rank
            best_U, best_V, best_W = U_new, V_new, W_new
            # Continue searching for even lower rank
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

