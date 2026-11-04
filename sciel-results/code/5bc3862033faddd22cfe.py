"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, least_squares

def _build_target_tensor(n, m, p):
    """Build the target tensor T[i,j,k] = 1 for all valid indices."""
    T = np.ones((n, m, p))
    return T

def _compute_error_fast(n, m, p, U, V, W):
    """Compute max error using tensor operations for speed."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _check_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Check if decomposition is valid."""
    return _compute_error_fast(n, m, p, U, V, W) < tol

def _try_rank_structured(n, m, p, rank, seed=42, max_restarts=3):
    """Try to find a valid decomposition with given rank using structured initialization."""
    np.random.seed(seed)
    
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        # Structured initialization: use block-diagonal-like patterns
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with random but structured values
        for c in range(rank):
            # Assign random patterns
            u_pattern = np.random.choice([-1, 0, 1], size=n*m, p=[0.3, 0.4, 0.3])
            v_pattern = np.random.choice([-1, 0, 1], size=m*p, p=[0.3, 0.4, 0.3])
            w_pattern = np.random.choice([-1, 0, 1], size=p*n, p=[0.3, 0.4, 0.3])
            U[:, c] = u_pattern
            V[:, c] = v_pattern
            W[:, c] = w_pattern
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Define residuals for least squares
        def residuals(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U_mat = U_flat.reshape(n*m, rank)
            V_mat = V_flat.reshape(m*p, rank)
            W_mat = W_flat.reshape(p*n, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                        res.append(val - 1.0)
            return np.array(res)
        
        try:
            result = least_squares(residuals, params, method='lm', max_nfev=200)
            err = np.max(np.abs(result.fun))
            if err < best_error:
                best_error = err
                best_params = result.x
            if best_error < 1e-10:
                break
        except Exception:
            continue
    
    if best_params is None:
        return None
    
    U_mat = best_params[:n*m*rank].reshape(n*m, rank)
    V_mat = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_mat = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    if _check_decomposition(n, m, p, U_mat, V_mat, W_mat):
        return U_mat, V_mat, W_mat
    
    return None

def _try_rank_random(n, m, p, rank, seed=42, max_restarts=2):
    """Try to find a valid decomposition with given rank using random initialization."""
    np.random.seed(seed)
    
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U_mat = U_flat.reshape(n*m, rank)
            V_mat = V_flat.reshape(m*p, rank)
            W_mat = W_flat.reshape(p*n, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                        res.append(val - 1.0)
            return np.array(res)
        
        try:
            result = least_squares(residuals, params, method='lm', max_nfev=150)
            err = np.max(np.abs(result.fun))
            if err < best_error:
                best_error = err
                best_params = result.x
            if best_error < 1e-10:
                break
        except Exception:
            continue
    
    if best_params is None:
        return None
    
    U_mat = best_params[:n*m*rank].reshape(n*m, rank)
    V_mat = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_mat = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    if _check_decomposition(n, m, p, U_mat, V_mat, W_mat):
        return U_mat, V_mat, W_mat
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Try ranks from 20 to schoolbook-1
    for target_rank in range(20, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Try structured initialization first
        result = _try_rank_structured(n, m, p, target_rank, seed=target_rank, max_restarts=2)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
        
        # If not found, try random initialization
        result = _try_rank_random(n, m, p, target_rank, seed=target_rank+100, max_restarts=1)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    # Convert to list format
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
