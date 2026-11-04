import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Uses numerical optimization to search for lower-rank decompositions.
    """
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with naive rank-24 decomposition
    rank_naive = nm * p  # 24
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_naive[a, r] = 1.0
                V_naive[b, r] = 1.0
                W_naive[c, r] = 1.0
    
    best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    best_rank = rank_naive
    
    # Try to find lower rank decompositions
    deadline = time.time() + 100  # Leave 20s buffer for safety
    
    for target_rank in range(10, rank_naive):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        n_restarts = 3 if target_rank >= 15 else 2
        found_valid = False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            np.random.seed(42 + restart)
            U_init = np.random.randn(nm, target_rank) * 0.5
            V_init = np.random.randn(mp, target_rank) * 0.5
            W_init = np.random.randn(pn, target_rank) * 0.5
            
            # Flatten for optimization
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                U = x[:nm*target_rank].reshape(nm, target_rank)
                V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten(x)
                # Compute tensor decomposition error
                # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # This is equivalent to: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
                error = 0.0
                for r in range(target_rank):
                    tensor_r = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                    error += np.sum((T - tensor_r)**2)
                # Actually we need sum over all r
                T_approx = np.zeros_like(T)
                for r in range(target_rank):
                    T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return np.sum((T - T_approx)**2)
            
            def gradient(x):
                U, V, W = unflatten(x)
                T_approx = np.zeros_like(T)
                for r in range(target_rank):
                    T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                diff = T_approx - T
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(target_rank):
                    # d/dU[a,r] sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                    grad_U[:,r] = np.einsum('abc,bc->a', diff, V[:,r][:,None]*W[:,r][None,:])
                    grad_V[:,r] = np.einsum('abc,ac->b', diff, U[:,r][:,None]*W[:,r][None,:])
                    grad_W[:,r] = np.einsum('abc,ab->c', diff, U[:,r][:,None]*V[:,r][None,:])
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            x0 = flatten(U_init, V_init, W_init)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
                
                if result.success or result.fun < 1e-10:
                    U_opt, V_opt, W_opt = unflatten(result.x)
                    
                    # Check if decomposition is exact
                    T_approx = np.zeros_like(T)
                    for r in range(target_rank):
                        T_approx += np.einsum('a,b,c->abc', U_opt[:,r], V_opt[:,r], W_opt[:,r])
                    
                    if np.max(np.abs(T - T_approx)) < 1e-8:
                        # Snap to rational values if possible
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        best_rank = target_rank
                        found_valid = True
                        break  # Move to next target_rank
                    
            except Exception as e:
                pass
        
        if found_valid:
            # Found a valid decomposition at this rank, try to go lower
            continue
        else:
            # If we can't find valid decomposition at this rank, try slightly higher
            # But for now, just continue searching
            pass
    
    return best_result
