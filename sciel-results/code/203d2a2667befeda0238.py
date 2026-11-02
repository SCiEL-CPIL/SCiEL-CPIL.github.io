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
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7
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

def _try_rank31_elimination(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition by eliminating one column and optimizing the rest."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank0 = U0.shape[1]
    
    # Precompute tensor entries
    entries_a = []
    entries_b = []
    entries_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries_a.append(a)
                entries_b.append(b)
                entries_c.append(c)
    
    entries_a = np.array(entries_a)
    entries_b = np.array(entries_b)
    entries_c = np.array(entries_c)
    N_entries = len(entries_a)
    
    # Try each column as the one to eliminate
    for target_col in range(rank0):
        if time.time() - start_time > time_budget * 0.85:
            break
        
        # Remove the target column
        U_cur = np.delete(U0, target_col, axis=1)
        V_cur = np.delete(V0, target_col, axis=1)
        W_cur = np.delete(W0, target_col, axis=1)
        
        rank = U_cur.shape[1]
        
        def objective(params):
            # Unpack parameters
            idx = 0
            U_flat = params[idx:idx + nm * rank]; idx += nm * rank
            V_flat = params[idx:idx + mp * rank]; idx += mp * rank
            W_flat = params[idx:idx + pn * rank]; idx += pn * rank
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            # Compute residuals
            UA = U[entries_a, :]
            VB = V[entries_b, :]
            WC = W[entries_c, :]
            pred = (UA * VB * WC).sum(axis=1)
            res = pred - 1.0
            
            # Sum of squared residuals
            return np.sum(res * res)
        
        def gradient(params):
            idx = 0
            U_flat = params[idx:idx + nm * rank]; idx += nm * rank
            V_flat = params[idx:idx + mp * rank]; idx += mp * rank
            W_flat = params[idx:idx + pn * rank]; idx += pn * rank
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            UA = U[entries_a, :]
            VB = V[entries_b, :]
            WC = W[entries_c, :]
            pred = (UA * VB * WC).sum(axis=1)
            res = pred - 1.0
            
            # Gradient of sum(res^2) is 2 * sum(res * d_pred/d_param)
            # d_pred/d_U[a,r] = V[b,r] * W[c,r]
            
            grad_U = np.zeros((nm, rank))
            grad_V = np.zeros((mp, rank))
            grad_W = np.zeros((pn, rank))
            
            # Vectorized accumulation
            # res shape (N,), VB shape (N, rank), WC shape (N, rank)
            # d_obj/d_U[entries_a[e], r] += 2 * res[e] * VB[e, r] * WC[e, r]
            
            dres_VW = 2.0 * res[:, None] * VB * WC  # (N, rank)
            np.add.at(grad_U, entries_a, dres_VW)
            
            dres_UW = 2.0 * res[:, None] * UA * WC  # (N, rank)
            np.add.at(grad_V, entries_b, dres_UW)
            
            dres_UV = 2.0 * res[:, None] * UA * VB  # (N, rank)
            np.add.at(grad_W, entries_c, dres_UV)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return grad
        
        # Pack initial parameters
        params0 = np.concatenate([U_cur.flatten(), V_cur.flatten(), W_cur.flatten()])
        
        # Run optimization
        try:
            result = minimize(
                objective,
                params0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 300, 'maxls': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < 1e-10:
                # Unpack result
                idx = 0
                U_flat = result.x[idx:idx + nm * rank]; idx += nm * rank
                V_flat = result.x[idx:idx + mp * rank]; idx += mp * rank
                W_flat = result.x[idx:idx + pn * rank]; idx += pn * rank
                
                U_res = U_flat.reshape(nm, rank)
                V_res = V_flat.reshape(mp, rank)
                W_res = W_flat.reshape(pn, rank)
                
                # Final verification
                UA = U_res[entries_a, :]
                VB = V_res[entries_b, :]
                WC = W_res[entries_c, :]
                pred = (UA * VB * WC).sum(axis=1)
                max_res = np.max(np.abs(pred - 1.0))
                
                if max_res < 1e-8:
                    return U_res, V_res, W_res
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the rank-32 solution
    U, V, W = _build_rank32()
    
    best_U, best_V, best_W = U, V, W
    best_rank = 32
    
    # Try to find a rank-31 decomposition
    if time.time() < deadline - 10:
        search_budget = min(120.0, deadline - time.time() - 5.0)
        result = _try_rank31_elimination(U, V, W, n, m, p, search_budget)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 31
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
