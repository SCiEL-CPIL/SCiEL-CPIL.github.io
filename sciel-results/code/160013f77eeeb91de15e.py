import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook-1 down to 20
    for rank in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline - 20:
            break
            
        result = _try_rank_fast(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            continue
    
    return best_result

def _try_rank_fast(n, m, p, rank, deadline):
    """
    Try to find a rank-rank decomposition using a smarter initialization strategy.
    We use the structure of the matrix multiplication tensor to create better initial guesses.
    """
    u_size = n * m * rank
    v_size = m * p * rank
    w_size = p * n * rank
    
    # Precompute the target tensor
    # T[i, k, j1, j2] = delta(j1, j2)
    delta = np.eye(m)
    
    def compute_residual(U, V, W):
        """Compute the residual for given factor matrices."""
        # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
        # Result should be (n, p, m, m) with delta(j1,j2)
        prod = np.einsum('ijr,j2kr,kinr->ikjj2', U, V, W)
        return prod - delta[np.newaxis, np.newaxis, :, :]
    
    def cost_function(params):
        U = params[:u_size].reshape(n, m, rank)
        V = params[u_size:u_size + v_size].reshape(m, p, rank)
        W = params[u_size + v_size:].reshape(p, n, rank)
        res = compute_residual(U, V, W)
        return np.sum(res ** 2)
    
    # Strategy 1: Use structured random initializations based on Kronecker-like patterns
    best_cost = float('inf')
    best_params = None
    
    num_attempts = 15
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + attempt * 137)
        
        # Different initialization strategies
        if attempt % 3 == 0:
            # Pure random
            U0 = np.random.randn(n, m, rank) * 0.5
            V0 = np.random.randn(m, p, rank) * 0.5
            W0 = np.random.randn(p, n, rank) * 0.5
        elif attempt % 3 == 1:
            # Structured: use Kronecker-like patterns
            U0 = np.zeros((n, m, rank))
            V0 = np.zeros((m, p, rank))
            W0 = np.zeros((p, n, rank))
            for r in range(rank):
                # Assign each rank-1 component to cover specific (i,j,k) tuples
                i_idx = r % n
                j_idx = (r // n) % m
                k_idx = (r // (n * m)) % p
                U0[i_idx, j_idx, r] = 1.0
                V0[j_idx, k_idx, r] = 1.0
                W0[k_idx, i_idx, r] = 1.0
                # Add some randomness
                U0[:, :, r] += np.random.randn(n, m) * 0.1
                V0[:, :, r] += np.random.randn(m, p) * 0.1
                W0[:, :, r] += np.random.randn(p, n) * 0.1
        else:
            # Hybrid: start from a partial schoolbook and perturb
            U0 = np.zeros((n, m, rank))
            V0 = np.zeros((m, p, rank))
            W0 = np.zeros((p, n, rank))
            # Fill with schoolbook pattern for first min(rank, n*m*p) components
            count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if count >= rank:
                            break
                        U0[i, j, count] = 1.0
                        V0[j, k, count] = 1.0
                        W0[k, i, count] = 1.0
                        count += 1
                    if count >= rank:
                        break
                if count >= rank:
                    break
            # Add randomness to remaining components
            for r in range(count, rank):
                U0[:, :, r] = np.random.randn(n, m) * 0.3
                V0[:, :, r] = np.random.randn(m, p) * 0.3
                W0[:, :, r] = np.random.randn(p, n) * 0.3
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Use Nelder-Mead or BFGS for local optimization
        from scipy.optimize import minimize
        
        try:
            result = minimize(cost_function, params0, method='BFGS', 
                            options={'maxiter': 50, 'gtol': 1e-6})
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Also try Nelder-Mead for a few attempts
        if attempt < 5:
            try:
                result2 = minimize(cost_function, params0, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                if result2.fun < best_cost:
                    best_cost = result2.fun
                    best_params = result2.x
            except Exception:
                pass
    
    if best_params is None:
        return None
    
    # Check if we found a good solution
    if best_cost > 1e-6:
        return None
    
    U_opt = best_params[:u_size].reshape(n, m, rank)
    V_opt = best_params[u_size:u_size + v_size].reshape(m, p, rank)
    W_opt = best_params[u_size + v_size:].reshape(p, n, rank)
    
    # Try to snap to half-integers
    for denom in [1, 2, 4, 8]:
        U_snap = np.round(U_opt * denom) / denom
        V_snap = np.round(V_opt * denom) / denom
        W_snap = np.round(W_opt * denom) / denom
        
        res = compute_residual(U_snap, V_snap, W_snap)
        max_err = np.max(np.abs(res))
        
        if max_err < 1e-9:
            U_out = U_snap.reshape(n * m, rank).tolist()
            V_out = V_snap.reshape(m * p, rank).tolist()
            W_out = W_snap.reshape(p * n, rank).tolist()
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return None
