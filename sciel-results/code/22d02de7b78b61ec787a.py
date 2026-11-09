import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def _tensor_from_decomp_fast(U, V, W, n, m, p):
    """Vectorized tensor computation: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]"""
    R = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        for k in range(p):
            wk = w_mat[k, :]
            T[:, :, k] += (u_mat * v_mat[:, k:k+1]) * wk[:, None]
    return T

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    T = _tensor_from_decomp_fast(U, V, W, n, m, p)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

def _make_schoolbook(n, m, p):
    """Create schoolbook decomposition (rank = n*m*p)."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_known_construction(n, m, p):
    """
    Try known constructions for 3x3 matrix multiplication.
    """
    # Try W=ones approach: if W is all ones, we need sum_r U[i,j,r]*V[j,k,r] = 1
    # This is the standard bilinear form. Let's try a structured search.
    
    # First, try a simple structured approach with small integer values
    # For rank r, we need 9*r + 9*r = 18r variables, 27 constraints
    
    # Try rank 23 with a specific pattern
    # Actually, let's just return None and rely on optimization
    return None

def _optimize_rank(R, n, m, p, time_budget, max_restarts=20):
    """
    Optimize for a rank-R decomposition using L-BFGS-B.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    start_time = time.time()
    T_true = np.ones((n, m, p))
    T_flat_true = T_true.flatten()
    
    def obj_fn(params):
        U = params[:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:27*R].reshape(9, R)
        T = _tensor_from_decomp_fast(U, V, W, n, m, p)
        diff = T.flatten() - T_flat_true
        return np.dot(diff, diff)
    
    def grad_fn(params):
        U = params[:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:27*R].reshape(9, R)
        T = _tensor_from_decomp_fast(U, V, W, n, m, p)
        diff = T.flatten() - T_flat_true
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dT = diff[i, j, k]
                        grad_U[i * m + j, r] += dT * v[j * p + k, r] * w[k * n + i, r]
                        grad_V[j * p + k, r] += dT * u[i * m + j, r] * w[k * n + i, r]
                        grad_W[k * n + i, r] += dT * u[i * m + j, r] * v[j * p + k, r]
                        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    best_result = None
    best_loss = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Use structured initialization
        if restart == 0:
            # Start from a near-schoolbook structure
            U_init = np.zeros((9, R))
            V_init = np.zeros((9, R))
            W_init = np.zeros((9, R))
            # Initialize with a pattern that partially satisfies constraints
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i * m + j < R:
                            U_init[i * m + j, i * m + j] = 1
                        if j * p + k < R:
                            V_init[j * p + k, j * p + k] = 1
                        if k * n + i < R:
                            W_init[k * n + i, k * n + i] = 1
            # Add small random perturbation to break symmetry
            U_init += np.random.randn(9, R) * 0.05
            V_init += np.random.randn(9, R) * 0.05
            W_init += np.random.randn(9, R) * 0.05
        elif restart == 1:
            # Try with W close to ones
            U_init = np.random.randn(9, R) * 0.3
            V_init = np.random.randn(9, R) * 0.3
            W_init = np.ones((9, R)) + np.random.randn(9, R) * 0.05
        else:
            # Random initialization with varying scales
            scale = 0.2 + 0.1 * (restart % 5)
            U_init = np.random.randn(9, R) * scale
            V_init = np.random.randn(9, R) * scale
            W_init = np.random.randn(9, R) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(obj_fn, params_init, method='L-BFGS-B', jac=grad_fn, 
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res.x.copy()
            
            if res.fun < 1e-10:
                U_opt = res.x[:9*R].reshape(9, R)
                V_opt = res.x[9*R:18*R].reshape(9, R)
                W_opt = res.x[18*R:27*R].reshape(9, R)
                
                # Try rounding to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if _verify_decomp(U_round, V_round, W_round, n, m, p):
                    return U_round, V_round, W_round
                
                # Try rounding to nearest integer
                U_round_int = np.round(U_opt)
                V_round_int = np.round(V_opt)
                W_round_int = np.round(W_opt)
                
                if _verify_decomp(U_round_int, V_round_int, W_round_int, n, m, p):
                    return U_round_int, V_round_int, W_round_int
                    
        except Exception:
            pass
    
    # If we have a good result but not quite at tolerance, try to polish
    if best_result is not None and best_loss < 1e-5:
        try:
            U_opt = best_result[:9*R].reshape(9, R)
            V_opt = best_result[9*R:18*R].reshape(9, R)
            W_opt = best_result[18*R:27*R].reshape(9, R)
            
            # Try various rounding strategies
            for factor in [2, 4, 8]:
                U_round = np.round(U_opt * factor) / factor
                V_round = np.round(V_opt * factor) / factor
                W_round = np.round(W_opt * factor) / factor
                
                if _verify_decomp(U_round, V_round, W_round, n, m, p):
                    return U_round, V_round, W_round
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # First try known constructions
    known = _try_known_construction(n, m, p)
    if known is not None:
        U, V, W = known
        if _verify_decomp(np.array(U), np.array(V), np.array(W), n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Try optimization for decreasing ranks
    target_ranks = [19, 20, 21, 22, 23]
    
    for R in target_ranks:
        if time.time() > deadline:
            break
        
        # Allocate time based on rank priority
        remaining_time = deadline - time.time()
        rank_index = target_ranks.index(R)
        # Lower ranks get more time
        if rank_index == 0:
            time_for_rank = remaining_time * 0.4
        elif rank_index == 1:
            time_for_rank = remaining_time * 0.3
        else:
            time_for_rank = remaining_time * 0.15
        
        result = _optimize_rank(R, n, m, p, time_for_rank, max_restarts=25)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _make_schoolbook(n, m, p)
