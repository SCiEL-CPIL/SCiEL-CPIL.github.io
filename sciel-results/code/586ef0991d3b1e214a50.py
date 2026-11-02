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

def _build_kronecker_init(n, m, p, r, rng):
    """
    Build initialization using Kronecker product structure.
    Since p = 2*3, we can think of the 3x6 multiplication as 
    (3x2) * (2x3) structure, allowing rank reduction.
    """
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    U = np.zeros((n_rows_u, r))
    V = np.zeros((n_rows_v, r))
    W = np.zeros((n_rows_w, r))
    
    # Use structured random initialization with Kronecker-like patterns
    # For each rank component, assign patterns based on block structure
    for col in range(r):
        # Random but structured assignment
        if col < n * m:
            # First block: identity-like in U
            i = col // m
            j = col % m
            U[i*m + j, col] = 1.0
            # Spread across V and W
            for k in range(p):
                V[j*p + k, col] = (1.0 / p)
            for k in range(p):
                W[k*n + i, col] = 1.0
        elif col < n * m * 2:
            # Second block: different pattern
            idx = col - n * m
            i = idx // m
            j = idx % m
            U[i*m + j, col] = 0.5
            for k in range(p):
                V[j*p + k, col] = (2.0 / p) * (1 if k % 2 == 0 else -1)
            for k in range(p):
                W[k*n + i, col] = 0.5 * (1 if k % 2 == 0 else -1)
        else:
            # Random fill for remaining columns
            U[:, col] = rng.randn(n_rows_u) * 0.1
            V[:, col] = rng.randn(n_rows_v) * 0.1
            W[:, col] = rng.randn(n_rows_w) * 0.1
    
    return U, V, W

def _objective_and_grad(params, n, m, p, r):
    """
    params: flattened [U (n*m x r), V (m*p x r), W (p*n x r)]
    Returns (loss, gradient)
    Loss = sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    """
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
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

def _try_rank_r(n, m, p, r, deadline, seed=0, n_attempts=8):
    """
    Attempt to find a valid decomposition with rank r using multiple strategies.
    """
    rng = np.random.RandomState(seed)
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    best_loss = np.inf
    best_params = None
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt == 0:
            # Kronecker-structured initialization
            U0, V0, W0 = _build_kronecker_init(n, m, p, r, rng)
        elif attempt == 1:
            # Identity-based initialization
            U0 = np.zeros((n_rows_u, r))
            V0 = np.zeros((n_rows_v, r))
            W0 = np.zeros((n_rows_w, r))
            for col in range(min(r, n*m*p)):
                i = col % n
                j = (col // n) % m
                k = (col // (n*m)) % p
                U0[i*m + j, col] = 1.0
                V0[j*p + k, col] = 1.0
                W0[k*n + i, col] = 1.0
        elif attempt == 2:
            # Small random
            scale = 0.05
            U0 = rng.randn(n_rows_u, r) * scale
            V0 = rng.randn(n_rows_v, r) * scale
            W0 = rng.randn(n_rows_w, r) * scale
        elif attempt == 3:
            # Medium random
            scale = 0.15
            U0 = rng.randn(n_rows_u, r) * scale
            V0 = rng.randn(n_rows_v, r) * scale
            W0 = rng.randn(n_rows_w, r) * scale
        elif attempt == 4:
            # Block diagonal structure
            U0 = np.zeros((n_rows_u, r))
            V0 = np.zeros((n_rows_v, r))
            W0 = np.zeros((n_rows_w, r))
            block_size = max(1, r // 6)
            for b in range(6):
                start = b * block_size
                end = min((b+1) * block_size, r)
                for col in range(start, end):
                    i = b % n
                    j = (b // n) % m
                    k = (b // (n*m)) % p
                    U0[i*m + j, col] = 1.0
                    V0[j*p + k, col] = 1.0
                    W0[k*n + i, col] = 1.0
        else:
            # Various random scales
            scale = 0.02 + 0.03 * (attempt - 5)
            U0 = rng.randn(n_rows_u, r) * scale
            V0 = rng.randn(n_rows_v, r) * scale
            W0 = rng.randn(n_rows_w, r) * scale
        
        params0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        
        # Use L-BFGS-B with tight tolerances
        try:
            res = minimize(_objective_and_grad, params0, args=(n, m, p, r),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-15,
                                   'maxfun': 5000})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
            
            if best_loss < 1e-14:
                break
        except Exception:
            continue
    
    if best_params is not None and best_loss < 1e-12:
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
                # Check if columns are proportional
                norm_a = np.linalg.norm(va)
                norm_b = np.linalg.norm(vb)
                if norm_a < 1e-10 or norm_b < 1e-10:
                    # Remove near-zero column
                    if norm_a < norm_b:
                        cols.pop(i)
                        U = np.delete(U, a, axis=1)
                        V = np.delete(V, a, axis=1)
                        W = np.delete(W, a, axis=1)
                        r -= 1
                        changed = True
                        continue
                    else:
                        cols.pop(j)
                        U = np.delete(U, b, axis=1)
                        V = np.delete(V, b, axis=1)
                        W = np.delete(W, b, axis=1)
                        r -= 1
                        changed = True
                        break
                else:
                    # Check if proportional
                    dot = np.dot(va, vb)
                    cos_sim = dot / (norm_a * norm_b)
                    if abs(cos_sim) > 0.9999:
                        # Merge: keep the one with larger norm
                        if norm_a >= norm_b:
                            cols.pop(j)
                            U = np.delete(U, b, axis=1)
                            V = np.delete(V, b, axis=1)
                            W = np.delete(W, b, axis=1)
                            r -= 1
                        else:
                            cols.pop(i)
                            U = np.delete(U, a, axis=1)
                            V = np.delete(V, a, axis=1)
                            W = np.delete(W, a, axis=1)
                            r -= 1
                            continue
                        changed = True
                        break
                j += 1
            if not changed:
                i += 1
    return U, V, W

def _polish(U, V, W, n, m, p, deadline):
    """
    Final high-precision polish pass with tighter tolerances.
    """
    r = U.shape[1]
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    try:
        res = minimize(_objective_and_grad, params, args=(n, m, p, r),
                      method='L-BFGS-B', jac=True,
                      options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-18,
                               'maxfun': 10000})
        
        if res.fun < 1e-16:
            U2 = res.x[:n_rows_u * r].reshape(n_rows_u, r)
            V2 = res.x[n_rows_u * r : n_rows_u * r + n_rows_v * r].reshape(n_rows_v, r)
            W2 = res.x[n_rows_u * r + n_rows_v * r :].reshape(n_rows_w, r)
            
            if _verify(U2, V2, W2, tol=1e-12):
                return U2, V2, W2
    except Exception:
        pass
    
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
    # Start from rank 30 and go down, trying more aggressive ranks later
    for r in range(30, 17, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds for robustness
        found = False
        for seed_offset in [0, 42, 123, 999]:
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_r(n, m, p, r, deadline - 5, seed=r + seed_offset, 
                                n_attempts=6 if r >= 25 else 4)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            # Try to reduce further via column merging
            if time.time() < deadline - 2:
                U2, V2, W2 = _try_reduce(best_U, best_V, best_W, deadline - 2)
                if _verify(U2, V2, W2):
                    best_U, best_V, best_W = U2, V2, W2
                    best_rank = best_U.shape[1]
            
            # Polish for higher precision
            if time.time() < deadline - 1:
                U3, V3, W3 = _polish(best_U, best_V, best_W, n, m, p, deadline - 1)
                if _verify(U3, V3, W3, tol=1e-12):
                    best_U, best_V, best_W = U3, V3, W3
            
            break
    
    # Final check: ensure we still have a valid decomposition
    if not _verify(best_U, best_V, best_W):
        # Fallback to original schoolbook
        best_U, best_V, best_W = _build_schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
