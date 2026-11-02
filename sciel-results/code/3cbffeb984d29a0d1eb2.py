import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, atol=1e-9):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    def try_rank_structured(r):
        """Try to find a rank-r decomposition using structured initialization
        based on the algebraic structure of matrix multiplication."""
        if time.time() > deadline:
            return None
        
        # For matrix multiplication, we can use a Strassen-like approach.
        # The key insight: we can parameterize the factors using the 
        # multiplicative structure of the indices.
        
        # Try different structured initializations
        initializations = []
        
        # Initialization 1: Based on schoolbook with random perturbations
        U_base = np.zeros((nm, r))
        V_base = np.zeros((mp, r))
        W_base = np.zeros((pn, r))
        
        # Distribute the 24 schoolbook terms across r columns
        terms = list(product(range(n), range(m), range(p)))
        for idx, (i, j, k) in enumerate(terms):
            col = idx % r
            U_base[i*m+j, col] += 1.0
            V_base[j*p+k, col] += 1.0
            W_base[k*n+i, col] += 1.0
        
        initializations.append((U_base.copy(), V_base.copy(), W_base.copy()))
        
        # Initialization 2: Random with sparsity (more restarts)
        for _ in range(10):
            U_rand = np.random.randn(nm, r) * 0.5
            V_rand = np.random.randn(mp, r) * 0.5
            W_rand = np.random.randn(pn, r) * 0.5
            # Apply sparsity
            mask_u = np.random.rand(nm, r) > 0.5
            mask_v = np.random.rand(mp, r) > 0.5
            mask_w = np.random.rand(pn, r) > 0.5
            U_rand[mask_u] = 0
            V_rand[mask_v] = 0
            W_rand[mask_w] = 0
            initializations.append((U_rand, V_rand, W_rand))
        
        # Initialization 3: Strassen-like structured patterns
        # For 2x3x4, we can try patterns that exploit the rectangular structure
        for pattern_seed in range(3):
            U_str = np.zeros((nm, r))
            V_str = np.zeros((mp, r))
            W_str = np.zeros((pn, r))
            
            # Create structured non-zero patterns
            for col in range(r):
                # Each column picks a subset of indices
                i_idx = (col * pattern_seed + 1) % n
                j_idx = (col * pattern_seed + 2) % m
                k_idx = (col * pattern_seed + 3) % p
                
                # Add multiple terms per column for better coverage
                for di in range(min(2, n - i_idx)):
                    for dj in range(min(2, m - j_idx)):
                        for dk in range(min(2, p - k_idx)):
                            ii = (i_idx + di) % n
                            jj = (j_idx + dj) % m
                            kk = (k_idx + dk) % p
                            val = 1.0 if (di + dj + dk) % 2 == 0 else -1.0
                            U_str[ii*m+jj, col] += val
                            V_str[jj*p+kk, col] += val
                            W_str[kk*n+ii, col] += val
            
            initializations.append((U_str, V_str, W_str))
        
        # Try each initialization with local optimization
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        from scipy.optimize import minimize
        
        for U0, V0, W0 in initializations:
            if time.time() > deadline - 15:
                break
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                return compute_error(U, V, W)
            
            def gradient(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                dU = np.einsum('abc,bk,ck->ak', diff, V, W)
                dV = np.einsum('abc,ak,ck->bk', diff, U, W)
                dW = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return grad
            
            try:
                # First pass: L-BFGS-B with more iterations
                res = minimize(objective, params0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500})
                
                err = res.fun
                if err < best_err:
                    best_err = err
                    U_opt = res.x[:r*nm].reshape(nm, r)
                    V_opt = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                    W_opt = res.x[r*(nm+mp):].reshape(pn, r)
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    
                    if err < 1e-8:
                        # Try rounding to half-integers
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        
                        if check_decomposition(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded
                        elif check_decomposition(U_opt, V_opt, W_opt):
                            return U_opt, V_opt, W_opt
                
                # Second pass: High-precision polish if close
                if best_err < 1e-4 and time.time() < deadline - 10:
                    params_polish = res.x.copy()
                    res2 = minimize(objective, params_polish, jac=gradient, method='L-BFGS-B', 
                                   options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800})
                    
                    err2 = res2.fun
                    if err2 < best_err:
                        best_err = err2
                        U_opt = res2.x[:r*nm].reshape(nm, r)
                        V_opt = res2.x[r*nm:r*(nm+mp)].reshape(mp, r)
                        W_opt = res2.x[r*(nm+mp):].reshape(pn, r)
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        
                        if err2 < 1e-8:
                            # Try multiple rounding strategies
                            for scale in [2, 4, 8]:
                                U_rounded = np.round(U_opt * scale) / scale
                                V_rounded = np.round(V_opt * scale) / scale
                                W_rounded = np.round(W_opt * scale) / scale
                                
                                if check_decomposition(U_rounded, V_rounded, W_rounded):
                                    return U_rounded, V_rounded, W_rounded
                            
                            if check_decomposition(U_opt, V_opt, W_opt):
                                return U_opt, V_opt, W_opt
            except Exception:
                pass
        
        # Final verification with best found
        if best_U is not None and best_err < 1e-6:
            # Try multiple rounding strategies
            for scale in [1, 2, 4, 8]:
                U_rounded = np.round(best_U * scale) / scale
                V_rounded = np.round(best_V * scale) / scale
                W_rounded = np.round(best_W * scale) / scale
                
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
            
            if check_decomposition(best_U, best_V, best_W):
                return best_U, best_V, best_W
        
        return None
    
    # Try ranks from low to high
    # The theoretical lower bound is max(nm, mp, pn) = 12
    # But for matrix multiplication, the rank is typically higher
    # Try from 16 to 23
    for rank in range(16, 24):
        if time.time() > deadline - 20:
            break
        res = try_rank_structured(rank)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
