import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _try_rank31_direct(n, m, p, U32, V32, W32, time_budget):
    """Try to find a rank-31 decomposition directly using sum-of-squares and smooth max (log-sum-exp) objectives."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = 31
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    # Support mask: indices where T_true == 1
    support_mask = (T_true == 1.0)
    
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Initialize with first 31 columns of rank-32 solution
    initializations = []
    initializations.append((U32[:, :rank].copy(), V32[:, :rank].copy(), W32[:, :rank].copy()))
    
    # Add random perturbations
    for i in range(8):
        U_rand = U32[:, :rank].copy() + np.random.randn(nm, rank) * 0.1
        V_rand = V32[:, :rank].copy() + np.random.randn(mp, rank) * 0.1
        W_rand = W32[:, :rank].copy() + np.random.randn(pn, rank) * 0.1
        initializations.append((U_rand, V_rand, W_rand))
    
    def compute_residuals(U, V, W):
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        R = M - T_true
        return R
    
    def compute_grads(R, U, V, W):
        """Compute gradients of sum of squared residuals."""
        # grad_U[i,r] = 2 * sum_jk R[i,j,k] V[j,r] W[k,r]
        A = np.einsum('ijk,kr->ijr', R, W)
        grad_U = 2.0 * np.einsum('ijr,jr->ir', A, V)
        
        # grad_V[j,r] = 2 * sum_ik R[i,j,k] U[i,r] W[k,r]
        B = np.einsum('ijk,kr->ijr', R, W)
        grad_V = 2.0 * np.einsum('ijr,ir->jr', B, U)
        
        # grad_W[k,r] = 2 * sum_ij R[i,j,k] U[i,r] V[j,r]
        D = np.einsum('ir,jr->ijr', U, V)
        grad_W = 2.0 * np.einsum('ijr,ijk->kr', D, R)
        
        return grad_U, grad_V, grad_W
    
    def objective_and_grad_sq(params):
        """Sum of squared residuals objective."""
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        R = compute_residuals(U, V, W)
        grad_U, grad_V, grad_W = compute_grads(R, U, V, W)
        
        sq_res = np.sum(R**2)
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return sq_res, grad
    
    def objective_and_grad_smooth_max(params, beta=20.0):
        """Smooth max (log-sum-exp) of squared residuals as epigraph surrogate."""
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        R = compute_residuals(U, V, W)
        R2 = R**2
        
        # Log-sum-exp for numerical stability
        R2_max = np.max(R2)
        if R2_max > 1e30:
            val = R2_max + np.log(np.sum(np.exp(R2 - R2_max))) / beta
            # Gradient scaled by beta * softmax
            weights = np.exp(beta * (R2 - R2_max))
            weights /= np.sum(weights)
            grad_R = 2.0 * R * weights
        else:
            val = np.log(np.sum(np.exp(beta * R2))) / beta
            weights = np.exp(beta * R2)
            weights /= np.sum(weights)
            grad_R = 2.0 * R * weights
        
        # Backpropagate to U, V, W
        # grad_U[i,r] = sum_jk grad_R[i,j,k] V[j,r] W[k,r]
        A = np.einsum('ijk,kr->ijr', grad_R, W)
        grad_U = np.einsum('ijr,jr->ir', A, V)
        
        B = np.einsum('ijk,kr->ijr', grad_R, W)
        grad_V = np.einsum('ijr,ir->jr', B, U)
        
        D = np.einsum('ir,jr->ijr', U, V)
        grad_W = np.einsum('ijr,ijk->kr', D, grad_R)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return val, grad
    
    for U0, V0, W0 in initializations:
        if time.time() - start_time > time_budget:
            break
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Phase 1: Sum of squares to get close
        try:
            result1 = minimize(objective_and_grad_sq, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000})
        except Exception:
            continue
        
        U_flat = result1.x[:nm * rank]
        V_flat = result1.x[nm * rank : nm * rank + mp * rank]
        W_flat = result1.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        res = _compute_residual(U, V, W, n, m, p)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U, V, W
        
        if res < 1e-9:
            return U, V, W, res
        
        # Phase 2: Smooth max to reduce max residual
        if time.time() - start_time > time_budget:
            break
        
        try:
            result2 = minimize(objective_and_grad_smooth_max, result1.x, jac=True, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000})
        except Exception:
            continue
        
        U_flat = result2.x[:nm * rank]
        V_flat = result2.x[nm * rank : nm * rank + mp * rank]
        W_flat = result2.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        res = _compute_residual(U, V, W, n, m, p)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U, V, W
        
        if res < 1e-9:
            return U, V, W, res
    
    if best_res < 1e-9:
        return best_U, best_V, best_W, best_res
    
    # Final high-precision polish: use the best found and do a tight optimization
    if best_U is not None and time.time() - start_time < time_budget * 0.8:
        x_polish = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
        try:
            result_polish = minimize(objective_and_grad_sq, x_polish, jac=True, method='L-BFGS-B',
                                     options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 10000})
            U_flat = result_polish.x[:nm * rank]
            V_flat = result_polish.x[nm * rank : nm * rank + mp * rank]
            W_flat = result_polish.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            res = _compute_residual(U, V, W, n, m, p)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U, V, W
        except Exception:
            pass
    
    if best_res < 1e-9:
        return best_U, best_V, best_W, best_res
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using direct optimization
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        search_budget = min(remaining_time - 20, 100)
        result = _try_rank31_direct(n, m, p, U32, V32, W32, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-9:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
                
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
