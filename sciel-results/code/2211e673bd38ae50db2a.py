import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try ranks from 22 to 24
    # Rank 22 is the theoretical upper bound from 2x3x3 (16) + 2x3x1 (6)
    # Rank 23 is also a good candidate
    # Rank 24 is schoolbook
    
    for rank in [23, 22, 24]:
        if time.time() > deadline - 20:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank(n, m, p, rank, deadline, max_iters=3000, tol=1e-12, n_restarts=40):
    """Try to find a rank-r decomposition via alternating minimization."""
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        success = _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol, deadline)
        if success:
            # Verify and clean up
            U, V, W = _clean_solution(n, m, p, rank, U, V, W)
            if _verify(n, m, p, U, V, W, tol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # If snap fails, try rounding to nearest integer?
            # Or just accept if error is tiny?
            # The prompt requires integer/half-integer.
            # If _verify fails after snapping, it's not a valid solution.
            # Try a different snap?
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify(n, m, p, U_int, V_int, W_int, tol=1e-10):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return None

def _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol, deadline):
    """Alternating least squares."""
    
    for it in range(max_iters):
        if time.time() > deadline - 20:
            return False
        
        # Solve for U
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except:
                    return False
        
        # Solve for V
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                except:
                    return False
        
        # Solve for W
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                except:
                    return False
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            return True
    
    return False

def _clean_solution(n, m, p, rank, U, V, W):
    """Snap to nearest half-integer."""
    U_clean = np.round(U * 2) / 2
    V_clean = np.round(V * 2) / 2
    W_clean = np.round(W * 2) / 2
    return U_clean, V_clean, W_clean

def _verify(n, m, p, U, V, W, tol=1e-10):
    """Verify exact decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
