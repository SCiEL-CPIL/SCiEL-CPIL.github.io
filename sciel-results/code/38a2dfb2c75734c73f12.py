"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact within tolerance."""
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res < tol, max_res

def _als_step(U, V, W, n, m, p, r):
    """One ALS iteration: fix two factors, solve linear system for the third."""
    new_U = U.copy()
    new_V = V.copy()
    new_W = W.copy()
    
    # Optimize U: for each (i,j), solve M * u = b where M[k,c] = V[j*p+k,c]*W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, r))
            for k in range(p):
                M[k, :] = V[j*p+k] * W[k*n+i]
            b = np.ones(p)
            try:
                u, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                new_U[i*m+j, :] = u
            except:
                pass
    
    # Optimize V: for each (j,k), solve M * v = b where M[i,c] = U[i*m+j,c]*W[k*n+i,c]
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, r))
            for i in range(n):
                M[i, :] = U[i*m+j] * W[k*n+i]
            b = np.ones(n)
            try:
                v, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                new_V[j*p+k, :] = v
            except:
                pass
    
    # Optimize W: for each (k,i), solve M * w = b where M[j,c] = U[i*m+j,c]*V[j*p+k,c]
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, r))
            for j in range(m):
                M[j, :] = U[i*m+j] * V[j*p+k]
            b = np.ones(m)
            try:
                w, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                new_W[k*n+i, :] = w
            except:
                pass
    
    return new_U, new_V, new_W

def _als_decompose(U_init, V_init, W_init, n, m, p, r, max_iter=200, tol=1e-14):
    """Run ALS to convergence."""
    U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
    prev_res = float('inf')
    
    for it in range(max_iter):
        U, V, W = _als_step(U, V, W, n, m, p, r)
        _, res = _check_exact(U, V, W, n, m, p, tol=1.0)
        if res < tol:
            break
        if abs(prev_res - res) < 1e-15 and it > 5:
            break
        prev_res = res
    
    return U, V, W

def _smooth_residual(U, V, W, n, m, p, beta=100.0):
    """Smooth surrogate: log-sum-exp of squared residuals (soft max)."""
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                residuals.append((val - 1.0) ** 2)
    residuals = np.array(residuals)
    # Soft max via log-sum-exp
    max_r = np.max(residuals)
    if max_r == 0:
        return 0.0
    return np.log(np.sum(np.exp(residuals - max_r))) + max_r

def _gradient_descent(U, V, W, n, m, p, r, max_iter=3000, lr_init=0.01, timeout=None):
    """Gradient descent on smooth surrogate objective."""
    start = time.time()
    lr = lr_init
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if timeout is not None and time.time() - start > timeout:
            break
        
        # Compute gradient of sum of squared residuals
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = 2.0 * (val - 1.0)
                    dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                    dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                    dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if it % 300 == 0 and it > 0:
            lr *= 0.7
        
        _, res = _check_exact(U, V, W, n, m, p, tol=1.0)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_res < 1e-12:
            break
    
    return best_res, best_U, best_V, best_W

def _try_rank(r, n, m, p, deadline):
    """Try to find exact rank-r decomposition using hybrid GD+ALS."""
    nm, mp, pn = n*m, m*p, p*n
    found = False
    
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.RandomState(12345 + r * 1000 + trial)
        scale = 1.0 / np.sqrt(r)
        U = rng.randn(nm, r) * scale
        V = rng.randn(mp, r) * scale
        W = rng.randn(pn, r) * scale
        
        # Phase 1: Gradient descent to get close
        timeout_gd = min(3.0, deadline - 20 - (time.time() - (deadline - 130)))
        if timeout_gd < 0.5:
            timeout_gd = 0.5
        _, U, V, W = _gradient_descent(U, V, W, n, m, p, r, max_iter=1000, lr_init=0.02, timeout=timeout_gd)
        
        # Phase 2: ALS polish to high precision
        U, V, W = _als_decompose(U, V, W, n, m, p, r, max_iter=500, tol=1e-15)
        
        # Final check
        exact, res = _check_exact(U, V, W, n, m, p, tol=1e-10)
        if exact:
            return True, U, V, W, res
    
    return False, None, None, None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Schoolbook fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    min_possible = max(nm, mp, pn)
    
    # Try ranks from lower bound up to schoolbook-1
    for r in range(min_possible, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        found, U, V, W, res = _try_rank(r, n, m, p, deadline)
        
        if found:
            best_rank = r
            best_U, best_V, best_W = U, V, W
            # Found exact, try to go lower
            continue
    
    # Convert to list format
    U_out = [[float(U_out[i][j]) for j in range(best_rank)] for i in range(nm)]
    V_out = [[float(V_out[i][j]) for j in range(best_rank)] for i in range(mp)]
    W_out = [[float(W_out[i][j]) for j in range(best_rank)] for i in range(pn)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
