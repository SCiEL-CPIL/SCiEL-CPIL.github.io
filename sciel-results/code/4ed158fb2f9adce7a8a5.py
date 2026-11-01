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

def _compute_residual_and_grad(U, V, W, T_true, target_col, lam):
    """Compute residual sum-of-squares and gradient for the objective."""
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    # M[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # grad_U[i,r] = sum_{j,k} (M[i,j,k] - T[i,j,k]) * V[j,r] * W[k,r]
    # grad_V[j,r] = sum_{i,k} (M[i,j,k] - T[i,j,k]) * U[i,r] * W[k,r]
    # grad_W[k,r] = sum_{i,j} (M[i,j,k] - T[i,j,k]) * U[i,r] * V[j,r]
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    R = M - T_true  # residual tensor
    
    # Gradient of sum(R^2) = 2 * sum R * dM/dparam
    # dM[i,j,k]/dU[i',r'] = delta(i,i')*V[j,r']*W[k,r']
    # So grad_U[i,r] = 2 * sum_{j,k} R[i,j,k] * V[j,r] * W[k,r]
    
    # Compute efficiently:
    # grad_U[i,r] = 2 * (R @ (V[r,:] * W[r,:]))[i] summed over j,k... 
    # Actually: grad_U[i,r] = 2 * sum_j sum_k R[i,j,k] * V[j,r] * W[k,r]
    # = 2 * sum_j (sum_k R[i,j,k]*W[k,r]) * V[j,r]
    # Let A[i,j,r] = sum_k R[i,j,k]*W[k,r] = einsum('ijk,kr->ijr', R, W)
    # Then grad_U[i,r] = 2 * sum_j A[i,j,r]*V[j,r] = 2 * einsum('ijr,jr->ir', A, V)
    
    A = np.einsum('ijk,kr->ijr', R, W)  # (nm, mp, rank)
    grad_U = 2.0 * np.einsum('ijr,jr->ir', A, V)  # (nm, rank)
    
    B = np.einsum('ijk,kr->ijr', R, W)  # same as A, reuse
    # grad_V[j,r] = 2 * sum_i sum_k R[i,j,k]*U[i,r]*W[k,r]
    # = 2 * sum_i (sum_k R[i,j,k]*W[k,r])*U[i,r]
    # = 2 * einsum('ijr,ir->jr', A, U)
    grad_V = 2.0 * np.einsum('ijr,ir->jr', A, U)  # (mp, rank)
    
    # grad_W[k,r] = 2 * sum_i sum_j R[i,j,k]*U[i,r]*V[j,r]
    # = 2 * einsum('ijk,ir,jr->kr', R, U, V)
    C = np.einsum('ijk,ir->jk', R, U)  # (mp, pn) -- wait, this isn't right
    # Let me redo: grad_W[k,r] = 2 * sum_i sum_j R[i,j,k]*U[i,r]*V[j,r]
    # = 2 * sum_i sum_j U[i,r]*V[j,r]*R[i,j,k]
    # Let D[i,j,r] = U[i,r]*V[j,r], then grad_W[k,r] = 2 * sum_i sum_j D[i,j,r]*R[i,j,k]
    # = 2 * einsum('ijr,ijk->kr', D, R)
    D = np.einsum('ir,jr->ijr', U, V)  # (nm, mp, rank)
    grad_W = 2.0 * np.einsum('ijr,ijk->kr', D, R)  # (pn, rank)
    
    # Add penalty gradient: penalty = lam * (sum U[:,tc]^2 + sum V[:,tc]^2 + sum W[:,tc]^2)
    # d(penalty)/dU[i,tc] = 2*lam*U[i,tc]
    grad_U[:, target_col] += 2.0 * lam * U[:, target_col]
    grad_V[:, target_col] += 2.0 * lam * V[:, target_col]
    grad_W[:, target_col] += 2.0 * lam * W[:, target_col]
    
    sq_res = np.sum(R**2)
    col_penalty = lam * (np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2))
    
    return sq_res + col_penalty, grad_U, grad_V, grad_W

def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition using gradient-based optimization."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank = 32
    
    # Precompute T_true
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    # Try each column as the one to eliminate
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.85:
            break
        
        # Try multiple lambda values and random perturbations
        best_col_norm = float('inf')
        best_result = None
        
        for lam in [0.01, 0.1, 0.5, 1.0, 5.0]:
            if time.time() - start_time > time_budget * 0.9:
                break
            
            # Try with small random perturbation to escape local minima
            for trial in range(3):
                if time.time() - start_time > time_budget * 0.9:
                    break
                
                # Perturb the target column slightly
                U_p = U0.copy()
                V_p = V0.copy()
                W_p = W0.copy()
                if trial > 0:
                    scale = 0.01 * trial
                    U_p[:, target_col] += np.random.randn(nm) * scale
                    V_p[:, target_col] += np.random.randn(mp) * scale
                    W_p[:, target_col] += np.random.randn(pn) * scale
                
                def objective_and_grad(params):
                    U_flat = params[:nm * rank]
                    V_flat = params[nm * rank : nm * rank + mp * rank]
                    W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                    
                    U = U_flat.reshape(nm, rank)
                    V = V_flat.reshape(mp, rank)
                    W = W_flat.reshape(pn, rank)
                    
                    val, gU, gV, gW = _compute_residual_and_grad(U, V, W, T_true, target_col, lam)
                    
                    grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
                    return val, grad
                
                x0 = np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()])
                
                try:
                    result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500})
                except Exception:
                    continue
                
                U_flat = result.x[:nm * rank]
                V_flat = result.x[nm * rank : nm * rank + mp * rank]
                W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                U = U_flat.reshape(nm, rank)
                V = V_flat.reshape(mp, rank)
                W = W_flat.reshape(pn, rank)
                
                col_norm = np.sqrt(np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2))
                
                if col_norm < best_col_norm:
                    best_col_norm = col_norm
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if col_norm < 0.01:
                    # Try rank-31 by dropping this column
                    U31 = np.delete(U, target_col, axis=1)
                    V31 = np.delete(V, target_col, axis=1)
                    W31 = np.delete(W, target_col, axis=1)
                    score31 = _compute_residual(U31, V31, W31, n, m, p)
                    if score31 < 1e-6:
                        return U31, V31, W31, score31
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using gradient-based local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        search_budget = min(remaining_time - 20, 80)
        result = _try_rank31_search(U, V, W, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-6:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Fall back to rank-32
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
