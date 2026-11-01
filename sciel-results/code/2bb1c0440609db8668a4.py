"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIMEOUT = 130  # seconds, safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _validate_decomposition(U, V, W, n, m, p):
    """Check if the tensor decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    return np.allclose(T, T_recon, atol=1e-9)

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen formulas for 2x2x2
    strassen_terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]
    
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx, col] += coeff
            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx, col] += coeff
    
    schoolbook_terms = [
        ([ (0, 1) ], [ (8, 1) ], [ (16, 1) ]),
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),
        ([ (2, 1) ], [ (8, 1) ], [ (17, 1) ]),
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),
    ]
    
    col_start = 4 * 7
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
    
    return U, V, W

def _residual_and_grad(x, U0, V0, W0, T_target):
    """
    Compute residual sum-of-squares and gradient for L-BFGS-B.
    x is a flattened vector containing [U_flat, V_flat, W_flat].
    """
    n, m, p = 2, 2, 9
    r = 31
    dim_U = n * m * r
    dim_V = m * p * r
    
    U = x[:dim_U].reshape(n * m, r)
    V = x[dim_U:dim_U + dim_V].reshape(m * p, r)
    W = x[dim_U + dim_V:].reshape(p * n, r)
    
    # Residual: T_target - sum_r U[:,r] V[:,r]^T W[:,r]
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    residual = T_target - T_recon
    loss = np.sum(residual ** 2)
    
    # Gradient computation
    # d_loss/dU[i,r] = -2 * sum_{j,k} residual[i,j,k] * V[j,r] * W[k,r]
    # d_loss/dV[j,r] = -2 * sum_{i,k} residual[i,j,k] * U[i,r] * W[k,r]
    # d_loss/dW[k,r] = -2 * sum_{i,j} residual[i,j,k] * U[i,r] * V[j,r]
    
    grad_U = -2.0 * np.einsum('ijk,jr,kr->ir', residual, V, W)
    grad_V = -2.0 * np.einsum('ijk,ir,kr->jr', residual, U, W)
    grad_W = -2.0 * np.einsum('ijk,ir,jr->kr', residual, U, V)
    
    grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
    
    return loss, grad

def _try_rank31_lbbfgs(n, m, p, removed_term, max_iter=200, time_limit=30):
    """
    Try to find a rank-31 decomposition by optimizing from the rank-32 solution
    with one term removed, using L-BFGS-B.
    """
    if time.time() - _START_TIME > _TIMEOUT - 5:
        return None
    
    U32, V32, W32 = _build_rank_32(n, m, p)
    T_target = _build_target_tensor(n, m, p)
    
    others = [i for i in range(32) if i != removed_term]
    U0 = U32[:, others].copy()
    V0 = V32[:, others].copy()
    W0 = W32[:, others].copy()
    
    # Initial loss
    T_recon0 = np.einsum('ir,jr,kr->ijk', U0, V0, W0)
    residual0 = T_target - T_recon0
    initial_loss = np.sum(residual0 ** 2)
    
    # Flatten initial point
    x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
    
    # Define objective and gradient
    def objective(x):
        loss, _ = _residual_and_grad(x, U0, V0, W0, T_target)
        return loss
    
    def gradient(x):
        _, grad = _residual_and_grad(x, U0, V0, W0, T_target)
        return grad
    
    # Time-limited optimization
    start_time = time.time()
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'maxfun': 500,
                'maxls': 50,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'iprint': 0,
            }
        )
    except Exception as e:
        return None
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
    
    if result.success or result.fun < 1e-8:
        x_opt = result.x
        dim_U = n * m * 31
        dim_V = m * p * 31
        U_opt = x_opt[:dim_U].reshape(n * m, 31)
        V_opt = x_opt[dim_U:dim_U + dim_V].reshape(m * p, 31)
        W_opt = x_opt[dim_U + dim_V:].reshape(p * n, 31)
        
        if _validate_decomposition(U_opt, V_opt, W_opt, n, m, p):
            return U_opt, V_opt, W_opt
    
    return None

def _search_rank31():
    """
    Try to find a rank-31 decomposition using L-BFGS-B optimization.
    """
    n, m, p = 2, 2, 9
    
    # First check if any column is exactly redundant
    U32, V32, W32 = _build_rank_32(n, m, p)
    for target in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 10:
            break
        others = [i for i in range(32) if i != target]
        U = U32[:, others]
        V = V32[:, others]
        W = W32[:, others]
        if _validate_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    # Try L-BFGS-B from each possible removed term
    # Prioritize terms from the schoolbook block (indices 28-31) as they may be more redundant
    priority_terms = [28, 29, 30, 31, 0, 7, 14, 21]
    remaining_terms = [i for i in range(32) if i not in priority_terms]
    all_terms = priority_terms + remaining_terms
    
    for removed in all_terms:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        
        result = _try_rank31_lbbfgs(n, m, p, removed, max_iter=150, time_limit=25)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-31 decomposition
    if time.time() - _START_TIME < _TIMEOUT - 15:
        result = _search_rank31()
        if result is not None:
            U, V, W = result
            if _validate_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-32 Strassen-based decomposition
    U, V, W = _build_rank_32(n, m, p)
    
    if _validate_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook rank 36
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
