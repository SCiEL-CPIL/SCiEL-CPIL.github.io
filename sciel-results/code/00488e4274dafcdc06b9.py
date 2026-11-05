"""Rank-minimal tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 8]

def _build_tensor():
    """Build the 3x3 x 3x8 multiplication tensor T[i,j,k]."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T (within tolerance)."""
    rank = U.shape[1]
    # Compute sum_c U[:,c] V[:,c] W[:,c]
    # Use einsum for efficiency
    result = np.einsum('ac,bc,dc->abd', U, V, W)
    diff = np.abs(result - T)
    return bool(np.max(diff) < tol)

def _try_rational_polish(U, V, W, T, max_iter=50):
    """Try to find exact rational entries close to the floating-point solution."""
    # For each matrix, try to snap to fractions with small denominators
    # This is a heuristic; we check if the snapped version still works
    U_int = np.round(U * 1000000) / 1000000.0
    V_int = np.round(V * 1000000) / 1000000.0
    W_int = np.round(W * 1000000) / 1000000.0
    if _verify_exact(U_int, V_int, W_int, T, tol=1e-7):
        return U_int, V_int, W_int
    return None

def _random_init(rank, n, m, p, rng):
    """Random initialization for U, V, W."""
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    return U, V, W

def _gradient_descent(U, V, W, T, max_iter=200, lr=0.01, tol=1e-10):
    """
    Gradient descent to minimize ||sum_c U_c V_c W_c - T||_F^2.
    Using alternating minimization for efficiency.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()

    for it in range(max_iter):
        # Fix V, W; optimize U
        # T[i,j,k] = sum_c U[i,c] * (V[j,c] * W[k,c])
        # For each c, V_c * W_c is a (m*p x p*n) matrix
        # U[:,c] should project T onto that
        # Residual approach:
        # T - sum_{c' != c} U[:,c'] V[:,c'] W[:,c'] = U[:,c] * (V[:,c] W[:,c])
        
        # Alternating: fix V and W, solve for U
        # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # For fixed V, W: T[i,:,:] = sum_c U[i,c] * (V[:,c] * W[:,c]).reshape(m*p, p*n)
        # This is a linear system: T_flat[i] = sum_c U[i,c] * M_c
        # where M_c = (V[:,c] * W[:,c]).flatten()
        
        # Build the (m*p * p*n) x rank matrix M
        # M[:, c] = (V[:,c] * W[:,c]).flatten()
        M = np.zeros((m * p * p * n, rank))
        for c in range(rank):
            M[:, c] = (V[:, c] * W[:, c]).reshape(-1)
        
        # T_flat[i, :] has shape (n*m, m*p*p*n)
        # T_flat[i, :] = U[i, :] @ M
        T_flat = T.reshape(n * m, m * p * p * n)
        
        # Solve U @ M = T_flat via least squares
        # U = T_flat @ M^+ 
        try:
            U_new = T_flat @ np.linalg.pinv(M)
        except np.linalg.LinAlgError:
            U_new = U
        
        U = U_new
        
        # Fix U, W; optimize V
        M2 = np.zeros((n * m * p * n, rank))
        for c in range(rank):
            M2[:, c] = (U[:, c] * W[:, c]).reshape(-1)
        T_flat2 = T.reshape(m * p, n * m * p * n)
        try:
            V_new = T_flat2 @ np.linalg.pinv(M2)
        except np.linalg.LinAlgError:
            V_new = V
        V = V_new
        
        # Fix U, V; optimize W
        M3 = np.zeros((n * m * m * p, rank))
        for c in range(rank):
            M3[:, c] = (U[:, c] * V[:, c]).reshape(-1)
        T_flat3 = T.reshape(p * n, n * m * m * p)
        try:
            W_new = T_flat3 @ np.linalg.pinv(M3)
        except np.linalg.LinAlgError:
            W_new = W
        W = W_new
        
        # Check convergence
        if it % 10 == 0:
            result = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(result - T, 'fro')
            if err < tol:
                break
    
    return U, V, W

def _search_rank(r, T, rng, n_restarts=10, max_iter=100, deadline=None):
    """Search for a rank-r decomposition with multiple random restarts."""
    n, m, p = DIMENSIONS
    best_err = np.inf
    best_U = best_V = best_W = None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        U, V, W = _random_init(r, n, m, p, rng)
        
        # Normalize to help convergence
        norms = np.linalg.norm(U, axis=0)
        norms[norms < 1e-10] = 1.0
        U = U / norms
        V = V * norms
        
        U, V, W = _gradient_descent(U, V, W, T, max_iter=max_iter)
        
        result = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.linalg.norm(result - T, 'fro')
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-9:
            break
    
    return best_err, best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    rng = np.random.default_rng(42)
    
    # The tensor has 9*24*24 = 5184 entries, but only 72 are nonzero.
    # Rank lower bound: at least max(n*m, m*p, p*n) = 24 for any decomposition
    # (since U has 9 rows, each rank-1 term contributes 1 to rank of U)
    # Actually rank >= max(9, 24, 24) = 24
    # Upper bound: schoolbook = 72
    
    # Try ranks from 24 upward
    best_result = None
    best_rank = n * m * p  # schoolbook as fallback
    
    # First, try to verify schoolbook works (sanity check)
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    
    if _verify_exact(U_s, V_s, W_s, T, tol=1e-9):
        best_result = (U_s, V_s, W_s)
        best_rank = rank_school
    
    # Try lower ranks
    for r in range(24, min(best_rank, 50)):
        if time.time() > deadline - 30:
            break
        
        # Adjust number of restarts based on time remaining
        time_left = deadline - time.time()
        n_restarts = max(2, min(10, int(time_left / 15)))
        
        err, U, V, W = _search_rank(r, T, rng, n_restarts=n_restarts, 
                                      max_iter=80, deadline=deadline)
        
        if err < 1e-8:
            # Try to get exact solution
            # The solution should have simple entries
            # Round to nearest 0.5 or integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify_exact(U_rounded, V_rounded, W_rounded, T, tol=1e-9):
                best_result = (U_rounded, V_rounded, W_rounded)
                best_rank = r
                break
            elif _verify_exact(U, V, W, T, tol=1e-10):
                best_result = (U, V, W)
                best_rank = r
                break
    
    if best_result is None:
        # Fallback to schoolbook
        best_result = (U_s, V_s, W_s)
        best_rank = rank_school
    
    U, V, W = best_result
    
    # Convert to lists for the required interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
