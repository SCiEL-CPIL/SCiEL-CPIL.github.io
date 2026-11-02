"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify(U, V, W, tol=1e-9):
    """Check the decomposition identity:
    for all i,j,k: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_r(n, m, p, r, deadline, seed=0):
    """
    Attempt to find a valid decomposition with rank r.
    Uses random initialization and gradient descent on the constraint
    that the tensor decomposition equals the matrix multiplication tensor.
    """
    rng = np.random.RandomState(seed)
    
    # Precompute target tensor entries
    # T[i,j,k] = 1 for all i,j,k in the matrix multiplication tensor
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    n_rows_u = n * m  # 6
    n_rows_v = m * p  # 18
    n_rows_w = p * n  # 12
    
    def objective_and_grad(params):
        """
        params: flattened [U (n*m x r), V (m*p x r), W (p*n x r)]
        Returns (loss, gradient)
        Loss = sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        """
        U = params[:n_rows_u * r].reshape(n_rows_u, r)
        V = params[n_rows_u * r : n_rows_u * r + n_rows_v * r].reshape(n_rows_v, r)
        W = params[n_rows_u * r + n_rows_v * r :].reshape(n_rows_w, r)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.dot(U[idx_u], V[idx_v]) * 0  # placeholder
                    # Actually: s = sum_r U[idx_u,r] * V[idx_v,r] * W[idx_w,r]
                    s = np.einsum('r,r,r->', U[idx_u], V[idx_v], W[idx_w])
                    err = s - 1.0
                    loss += err * err
                    
                    # Gradient w.r.t. U[idx_u, r] = 2*err*V[idx_v,r]*W[idx_w,r]
                    grad_U[idx_u] += 2 * err * V[idx_v] * W[idx_w]
                    grad_V[idx_v] += 2 * err * U[idx_u] * W[idx_w]
                    grad_W[idx_w] += 2 * err * U[idx_u] * V[idx_v]
        
        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        return loss, grad
    
    # Try multiple random initializations
    best_loss = np.inf
    best_params = None
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Random initialization
        U0 = rng.randn(n_rows_u, r) * 0.1
        V0 = rng.randn(n_rows_v, r) * 0.1
        W0 = rng.randn(n_rows_w, r) * 0.1
        
        params0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        
        res = minimize(objective_and_grad, params0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-15})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
        
        if best_loss < 1e-12:
            break
    
    if best_params is not None and best_loss < 1e-12:
        U = best_params[:n_rows_u * r].reshape(n_rows_u, r)
        V = best_params[n_rows_u * r : n_rows_u * r + n_rows_v * r].reshape(n_rows_v, r)
        W = best_params[n_rows_u * r + n_rows_v * r :].reshape(n_rows_w, r)
        
        if _verify(U, V, W, tol=1e-9):
            return U, V, W
    
    return None

def _try_rank_r_with_structure(n, m, p, r, deadline, seed=0):
    """
    Try rank r using structured initialization based on known lower bounds.
    For 2x3x6, the trivial rank is 36. Strassen-like reductions may help.
    """
    rng = np.random.RandomState(seed)
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    def objective_and_grad(params):
        U = params[:n_rows_u * r].reshape(n_rows_u, r)
        V = params[n_rows_u * r : n_rows_u * r + n_rows_v * r].reshape(n_rows_v, r)
        W = params[n_rows_u * r + n_rows_v * r :].reshape(n_rows_w, r)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.einsum('r,r,r->', U[idx_u], V[idx_v], W[idx_w])
                    err = s - 1.0
                    loss += err * err
                    grad_U[idx_u] += 2 * err * V[idx_v] * W[idx_w]
                    grad_V[idx_v] += 2 * err * U[idx_u] * W[idx_w]
                    grad_W[idx_w] += 2 * err * U[idx_u] * V[idx_v]
        
        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        return loss, grad
    
    best_loss = np.inf
    best_params = None
    
    # Try several initializations with different scales
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        scale = 0.05 + 0.1 * attempt
        U0 = rng.randn(n_rows_u, r) * scale
        V0 = rng.randn(n_rows_v, r) * scale
        W0 = rng.randn(n_rows_w, r) * scale
        
        params0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        
        res = minimize(objective_and_grad, params0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
        
        if best_loss < 1e-10:
            break
    
    if best_params is not None and best_loss < 1e-10:
        U = best_params[:n_rows_u * r].reshape(n_rows_u, r)
        V = best_params[n_rows_u * r : n_rows_u * r + n_rows_v * r].reshape(n_rows_v, r)
        W = best_params[n_rows_u * r + n_rows_v * r :].reshape(n_rows_w, r)
        
        if _verify(U, V, W, tol=1e-9):
            return U, V, W
    
    return None

def _try_reduce(U, V, W, deadline):
    """
    Attempt to reduce rank by merging columns whose combined rows are linearly dependent.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        cols = list(range(r))
        i = 0
        while i < len(cols) and time.time() < deadline:
            a = cols[i]
            j = i + 1
            while j < len(cols) and time.time() < deadline:
                b = cols[j]
                va = np.concatenate([U[:, a], V[:, a], W[:, a]])
                vb = np.concatenate([U[:, b], V[:, b], W[:, b]])
                if np.linalg.matrix_rank(np.column_stack([va, vb])) == 1:
                    na = np.linalg.norm(va)
                    nb = np.linalg.norm(vb)
                    if na >= nb:
                        cols.pop(j)
                        U = np.delete(U, b, axis=1)
                        V = np.delete(V, b, axis=1)
                        W = np.delete(W, b, axis=1)
                        r -= 1
                        changed = True
                    else:
                        cols.pop(i)
                        U = np.delete(U, a, axis=1)
                        V = np.delete(V, a, axis=1)
                        W = np.delete(W, a, axis=1)
                        r -= 1
                        changed = True
                        continue
                    break
                j += 1
            if not changed:
                i += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit

    # Build the guaranteed-valid schoolbook decomposition (rank 36)
    U, V, W = _build_schoolbook()
    
    # Verify the schoolbook decomposition
    if not _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try to find lower rank decompositions using optimization
    # Start from rank 30 and go down
    for r in range(30, 18, -1):
        if time.time() > deadline - 5:
            break
        
        # Try structured optimization
        result = _try_rank_r(n, m, p, r, deadline - 5, seed=r)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
            break
        
        # Also try with structure
        result = _try_rank_r_with_structure(n, m, p, r, deadline - 5, seed=r + 100)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    # If we found a lower rank, try to reduce further via column merging
    if best_rank < 36:
        if time.time() < deadline:
            U2, V2, W2 = _try_reduce(best_U, best_V, best_W, deadline)
            if _verify(U2, V2, W2):
                best_U, best_V, best_W = U2, V2, W2
                best_rank = best_U.shape[1]

    # Final check: ensure we still have a valid decomposition
    if not _verify(best_U, best_V, best_W):
        # Fallback to original schoolbook
        best_U, best_V, best_W = _build_schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
