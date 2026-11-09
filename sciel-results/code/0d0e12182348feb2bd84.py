import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Try ranks from lower to upper
    # Start from a reasonable lower bound
    # The naive rank is n*m*p = 50
    # Try from 30 up to 50
    
    for rank in range(30, 51):
        if time.time() > deadline:
            break
        
        result = _search_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: naive rank-50 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    max_restarts = 5
    max_iters = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(restart * 1000)
        
        # Random initialization
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for W given U, V
            W = _solve_W(n, m, p, U, V, rank)
            # Solve for U given V, W
            U = _solve_U(n, m, p, V, W, rank)
            # Solve for V given U, W
            V = _solve_V(n, m, p, U, W, rank)
            
            # Compute error
            err = _compute_error(n, m, p, U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-8:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            # Small perturbation
            U += 0.01 * np.random.randn(dim_u, rank)
            V += 0.01 * np.random.randn(dim_v, rank)
            W += 0.01 * np.random.randn(dim_w, rank)
        
        if best_err < 1e-8:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _solve_W(n, m, p, U, V, rank):
    """Solve for W given U, V."""
    dim_w = p * n
    W = np.zeros((dim_w, rank))
    for w_idx in range(dim_w):
        i = w_idx % n
        k = w_idx // n
        A = np.zeros((m, rank))
        for j in range(m):
            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
        try:
            W[w_idx, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
        except:
            W[w_idx, :] = 0
    return W

def _solve_U(n, m, p, V, W, rank):
    """Solve for U given V, W."""
    dim_u = n * m
    U = np.zeros((dim_u, rank))
    for u_idx in range(dim_u):
        i = u_idx // m
        j = u_idx % m
        B = np.zeros((p, rank))
        for k in range(p):
            B[k, :] = V[j*p+k, :] * W[k*n+i, :]
        try:
            U[u_idx, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
        except:
            U[u_idx, :] = 0
    return U

def _solve_V(n, m, p, U, W, rank):
    """Solve for V given U, W."""
    dim_v = m * p
    V = np.zeros((dim_v, rank))
    for v_idx in range(dim_v):
        j = v_idx // p
        k = v_idx % p
        C = np.zeros((n, rank))
        for i in range(n):
            C[i, :] = U[i*m+j, :] * W[k*n+i, :]
        try:
            V[v_idx, :] = np.linalg.lstsq(C, np.ones(n), rcond=None)[0]
        except:
            V[v_idx, :] = 0
    return V

def _compute_error(n, m, p, U, V, W):
    """Compute the error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return np.sqrt(err)
# EVOLVE-BLOCK-END
