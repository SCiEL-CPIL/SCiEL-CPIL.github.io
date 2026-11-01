import numpy as np
import time
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
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
        # U: (nm, r), V: (mp, r), W: (pn, r)
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    def try_round_to_half_integers(U, V, W):
        """Try rounding to nearest half-integers and verify exactness."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if check_decomposition(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        # Try rounding to nearest integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        if check_decomposition(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try mixed rounding strategies
        for scale in [1, 2, 4]:
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if check_decomposition(U_s, V_s, W_s):
                return U_s, V_s, W_s
        
        return None
    
    def try_rank_with_structure(r, num_restarts=50):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        # Number of variables: r * (nm + mp + pn)
        size = r * (nm + mp + pn)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            U_init = np.zeros((nm, r))
            V_init = np.zeros((mp, r))
            W_init = np.zeros((pn, r))
            
            # Different initialization strategies
            strategy = restart % 6
            
            if strategy == 0:
                # Pure random
                params = np.random.randn(size) * 0.5
            elif strategy == 1:
                # Structured: schoolbook-like with noise
                for col in range(min(r, nm*p)):
                    i = col // (m*p)
                    j = (col // p) % m
                    k = col % p
                    if i < n and j < m and k < p:
                        U_init[i*m+j, col] = 1.0
                        V_init[j*p+k, col] = 1.0
                        W_init[k*n+i, col] = 1.0
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                params += np.random.randn(size) * 0.2
            elif strategy == 2:
                # Sparse random with larger values
                params = np.random.randn(size)
                mask = np.random.rand(size) > 0.6
                params[mask] = 0
                params *= 1.0
            elif strategy == 3:
                # Block structure with random assignments
                for col in range(r):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U_init[i*m+j, col] = 1.0
                    V_init[j*p+k, col] = 1.0
                    W_init[k*n+i, col] = 1.0
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                params += np.random.randn(size) * 0.1
            elif strategy == 4:
                # Symmetric initialization
                for col in range(r):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U_init[i*m+j, col] = 1.0
                    V_init[j*p+k, col] = 1.0
                    W_init[k*n+i, col] = 1.0
                    # Add symmetry: also set related entries
                    if i+1 < n:
                        U_init[(i+1)*m+j, col] = 0.5
                    if j+1 < m:
                        V_init[(j+1)*p+k, col] = 0.5
                    if k+1 < p:
                        W_init[(k+1)*n+i, col] = 0.5
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Random with sign patterns
                params = np.random.choice([-1, 0, 1], size=size) * 0.5
                # Ensure some non-zero entries
                for _ in range(size // 4):
                    idx = np.random.randint(0, size)
                    params[idx] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            
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
            res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 2000})
            
            if time.time() > deadline:
                break
            
            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                
                if err < 1e-10:
                    U = res.x[:r*nm].reshape(nm, r)
                    V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                    W = res.x[r*(nm+mp):].reshape(pn, r)
                    
                    # Try rounding to half-integers
                    rounded = try_round_to_half_integers(U, V, W)
                    if rounded is not None:
                        return rounded
                    elif check_decomposition(U, V, W):
                        return U, V, W
        
        return None
    
    # Try ranks from 22 to 23
    for rank in [22, 23]:
        if time.time() > deadline - 30:
            break
        res = try_rank_with_structure(rank, num_restarts=60)
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
