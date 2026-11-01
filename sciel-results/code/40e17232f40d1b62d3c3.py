import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Flatten target for least_squares
    target_flat = target.flatten()

    def compute_residuals(params, r):
        """
        params: flattened U, V, W
        r: rank
        """
        U = params[:r * nm].reshape(nm, r)
        V = params[r * nm:r * (nm + mp)].reshape(mp, r)
        W = params[r * (nm + mp):].reshape(pn, r)
        
        # Compute T_approx
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return (T_approx - target).flatten()

    def try_rank(r, num_restarts=100):
        if time.time() > deadline:
            return None
        
        size = r * (nm + mp + pn)
        best_res = None
        best_cost = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialization
            if restart % 3 == 0:
                # Random
                params = np.random.randn(size) * 0.5
            elif restart % 3 == 1:
                # Sparse random
                params = np.random.randn(size)
                params[np.abs(params) < 0.5] = 0
            else:
                # Structured: Start with schoolbook (padded) and perturb
                U_init = np.zeros((nm, r))
                V_init = np.zeros((mp, r))
                W_init = np.zeros((pn, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i * m * p + j * p + k
                            if col < r:
                                U_init[i * m + j, col] = 1.0
                                V_init[j * p + k, col] = 1.0
                                W_init[k * n + i, col] = 1.0
                # Perturb
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                params += np.random.randn(size) * 0.1

            try:
                res = least_squares(compute_residuals, params, args=(r,), method='lm', max_nfev=1000)
            except Exception:
                continue

            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res

            if best_cost < 1e-10:
                # Check if we can round to integer/half-integer
                U = best_res.x[:r * nm].reshape(nm, r)
                V = best_res.x[r * nm:r * (nm + mp)].reshape(mp, r)
                W = best_res.x[r * (nm + mp):].reshape(pn, r)
                
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded solution
                T_approx = np.einsum('ak,bk,ck->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_approx, target, atol=1e-9):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # Verify unrounded
                    T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                    if np.allclose(T_approx, target, atol=1e-9):
                        return U, V, W

        # If we found a very low cost but didn't return inside the loop (e.g. rounding failed), 
        # try to round the best found solution one last time
        if best_res is not None and best_cost < 1e-8:
            U = best_res.x[:r * nm].reshape(nm, r)
            V = best_res.x[r * nm:r * (nm + mp)].reshape(mp, r)
            W = best_res.x[r * (nm + mp):].reshape(pn, r)
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_approx = np.einsum('ak,bk,ck->abc', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_approx, target, atol=1e-9):
                return U_rounded, V_rounded, W_rounded

        return None

    # Try rank 23
    res = try_rank(23, num_restarts=200)
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
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
