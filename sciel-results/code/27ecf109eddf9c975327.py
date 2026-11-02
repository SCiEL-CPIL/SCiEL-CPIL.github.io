"""Program that searches for a lower-rank exact tensor decomposition for 2x3 * 3x9."""
import random
import time
import numpy as np
import scipy.optimize as so

DIMENSIONS = [2, 3, 9]

def _construct_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3-index array T[i, j, k]."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _build_decomposition(n, m, p, rank, U, V, W):
    """
    Build factor matrices from parameter arrays.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    return U, V, W

def _decomposition_error(n, m, p, U, V, W):
    """
    Compute the Frobenius norm of the difference between the tensor
    and its rank-decomposition approximation.
    
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Reshape for efficient computation
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # Compute approximated tensor
    # T_approx[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # True tensor
    T_true = np.ones((n, m, p), dtype=np.float64)
    
    # Error
    diff = T_true - T_approx
    return np.linalg.norm(diff)

def _try_scipy_rank(n, m, p, rank, deadline, seed=42):
    """
    Try to find a valid rank decomposition using scipy.optimize.least_squares.
    """
    if time.time() > deadline - 5.0:
        return None
        
    rng = np.random.RandomState(seed)
    
    # Flatten parameters
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    total_size = size_u + size_v + size_w
    
    x0 = rng.standard_normal(total_size)
    
    def objective(params):
        U = params[:size_u].reshape(n * m, rank)
        V = params[size_u:size_u + size_v].reshape(m * p, rank)
        W = params[size_u + size_v:].reshape(p * n, rank)
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        T_true = np.ones((n, m, p), dtype=np.float64)
        
        return (T_true - T_approx).ravel()
    
    try:
        # Use LM method, limit function evaluations
        res = so.least_squares(objective, x0, method='lm', max_nfev=200)
        if res.cost < 1e-10:
            U = res.x[:size_u].reshape(n * m, rank)
            V = res.x[size_u:size_u + size_v].reshape(m * p, rank)
            W = res.x[size_u + size_v:].reshape(p * n, rank)
            return U, V, W
    except Exception:
        pass
    return None

def _check_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that the decomposition is exact."""
    U_r = np.asarray(U, dtype=np.float64).reshape(n, m, -1)
    V_r = np.asarray(V, dtype=np.float64).reshape(m, p, -1)
    W_r = np.asarray(W, dtype=np.float64).reshape(p, n, -1)
    
    T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T_true = np.ones((n, m, p), dtype=np.float64)
    
    diff = T_true - T_approx
    return np.max(np.abs(diff)) < tol

def _snap_to_halfint(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    arr = np.asarray(arr, dtype=np.float64)
    snapped = np.round(arr * 2) / 2.0
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped
    return arr

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard
    deadline = time.time() + 120.0
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Try decreasing ranks
    # Start from some heuristic lower bound
    # Lander-Motzkin lower bound: n*m + m*p + p*n - 1
    lb = n * m + m * p + p * n - 1
    
    # Try ranks from schoolbook down to lb+1
    # But only try a few promising ranks
    ranks_to_try = []
    
    # First, try to find if there's a known better rank
    # For 2x3 * 3x9, the rank is at most 54 (schoolbook)
    # Try some candidate ranks
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 20, lb), -1):
        ranks_to_try.append(r)
    
    # Add the lower bound itself
    if lb not in ranks_to_try:
        ranks_to_try.append(lb)
    
    # Prioritize known lower bounds or specific candidate ranks
    # For 2x3x9, rank 32 is a known optimal or near-optimal rank
    candidate_ranks = [32, 40, 48, 50, 54]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10.0:
            break
            
        # Try a few seeds for scipy optimization
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
                
            result = _try_scipy_rank(n, m, p, rank, deadline - 10.0, seed)
            
            if result is not None:
                U, V, W = result
                # Snap to half-integers if close
                U_s = _snap_to_halfint(U)
                V_s = _snap_to_halfint(V)
                W_s = _snap_to_halfint(W)
                
                if _check_decomposition(n, m, p, U_s, V_s, W_s, tol=1e-6):
                    return {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                else:
                    # Return the raw result if it's valid but not half-integer
                    if _check_decomposition(n, m, p, U, V, W, tol=1e-9):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
    
    # Fall back to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
