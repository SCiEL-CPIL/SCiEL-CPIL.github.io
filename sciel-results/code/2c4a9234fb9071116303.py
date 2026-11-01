"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _check_exact(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and compare to the standard tensor: T_std[i,j,k] = 1 if this is a valid product term.
    # Actually, the matrix multiplication tensor for (n x m) * (m x p) -> (n x p):
    # The tensor T has indices (i, k, j) where i in [0,n), k in [0,p), j in [0,m)
    # T[i,k,j] = 1 for all valid i,k,j (since C[i,k] = sum_j A[i,j]*B[j,k])
    # But the factorization uses U indexed by (i,j), V by (j,k), W by (k,i)
    # So the tensor we're decomposing is: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # This should equal 1 for all (i,j,k) since each term in the sum over c picks out
    # exactly one (i,j,k) combination in the schoolbook case.
    # More precisely, the target tensor T[i,j,k] = 1 for all i,j,k.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True


def _try_rank(r, n, m, p, time_budget):
    """Try to find a rank-r decomposition via random initialization and optimization."""
    U0 = np.random.randn(n * m, r)
    V0 = np.random.randn(m * p, r)
    W0 = np.random.randn(p * n, r)
    
    # Scale to encourage small values
    U0 *= 0.5
    V0 *= 0.5
    W0 *= 0.5
    
    def objective(params):
        U = params[:n * m * r].reshape(n * m, r)
        V = params[n * m * r:n * m * r + m * p * r].reshape(m * p, r)
        W = params[n * m * r + m * p * r:].reshape(p * n, r)
        
        # Compute the tensor error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (val - 1.0) ** 2
        return err
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    best_err = objective(x0)
    best_params = x0.copy()
    
    start = time.time()
    for attempt in range(3):
        if time.time() - start > time_budget:
            break
        result = minimize(objective, x0, method='BFGS', 
                         options={'maxiter': 100, 'gtol': 1e-4})
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x.copy()
        x0 = result.x + np.random.randn(len(result.x)) * 0.1
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        U = best_params[:n * m * r].reshape(n * m, r)
        V = best_params[n * m * r:n * m * r + m * p * r].reshape(m * p, r)
        W = best_params[n * m * r + m * p * r:].reshape(p * n, r)
        # Round to nearest half-integer and check
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        if _check_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-8):
            return U_rounded, V_rounded, W_rounded
        # Also check without rounding
        if _check_exact(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try ranks from best_rank - 1 downward
    for r in range(best_rank - 1, max(n * m, m * p, p * n) - 1, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        result = _try_rank(r, n, m, p, time_budget=remaining - 5)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
