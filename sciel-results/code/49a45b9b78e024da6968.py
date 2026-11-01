import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
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
        # U: (nm, r), V: (mp, r), W: (pn, r)
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    def schoolbook_init(r):
        """Initialize with schoolbook decomposition padded to rank r."""
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    if col < r:
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
        return U, V, W
    
    def try_rank_with_optimization(r, num_restarts=30):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        # Number of variables: r * (nm + mp + pn)
        size = r * (nm + mp + pn)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart % 4 == 0:
                # Pure random
                params = np.random.randn(size) * 0.3
            elif restart % 4 == 1:
                # Schoolbook + noise
                U_init, V_init, W_init = schoolbook_init(r)
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                params += np.random.randn(size) * 0.05
            elif restart % 4 == 2:
                # Sparse random
                params = np.random.randn(size)
                mask = np.random.rand(size) > 0.7
                params[mask] = 0
                params *= 0.5
            else:
                # Structured with partial schoolbook
                U_init, V_init, W_init = schoolbook_init(r)
                # Add some random structure
                for _ in range(min(5, r)):
                    col = np.random.randint(r)
                    U_init[:, col] += np.random.randn(nm) * 0.2
                    V_init[:, col] += np.random.randn(mp) * 0.2
                    W_init[:, col] += np.random.randn(pn) * 0.2
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
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
            
            # Use L-BFGS-B with gradient
            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 1000})
            except Exception:
                continue
            
            if time.time() > deadline - 5:
                break
            
            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                
                if err < 1e-10:
                    U = res.x[:r*nm].reshape(nm, r)
                    V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                    W = res.x[r*(nm+mp):].reshape(pn, r)
                    
                    # Try rounding to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded):
                        return U_rounded, V_rounded, W_rounded
                    elif check_decomposition(U, V, W):
                        return U, V, W
        
        return None
    
    def try_rank_with_differential_evolution(r, max_iter=50):
        if time.time() > deadline:
            return None
        
        size = r * (nm + mp + pn)
        
        def objective(params):
            U = params[:r*nm].reshape(nm, r)
            V = params[r*nm:r*(nm+mp)].reshape(mp, r)
            W = params[r*(nm+mp):].reshape(pn, r)
            return compute_error(U, V, W)
        
        # Bounds for differential evolution
        bounds = [(-2, 2)] * size
        
        try:
            res = differential_evolution(objective, bounds, maxiter=max_iter, 
                                         tol=1e-12, seed=42, polish=True)
        except Exception:
            return None
        
        if time.time() > deadline - 5:
            return None
        
        if res.fun < 1e-10:
            U = res.x[:r*nm].reshape(nm, r)
            V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
            W = res.x[r*(nm+mp):].reshape(pn, r)
            
            # Try rounding to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
            elif check_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    # Try rank 23 first, then 22
    for rank in [23, 22]:
        if time.time() > deadline - 30:
            break
        
        # First try L-BFGS-B optimization
        res = try_rank_with_optimization(rank, num_restarts=20)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If L-BFGS-B fails, try differential evolution (slower but more global)
        if time.time() > deadline - 10:
            break
        res = try_rank_with_differential_evolution(rank, max_iter=30)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24
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
# EVOLVE-BLOCK-END
