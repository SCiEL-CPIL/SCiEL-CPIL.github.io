import scipy
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = nm * mp * pn // (n * m * p) * (n * m * p) # Just n*m*p
    rank_schoolbook = n * m * p
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    best_U = None
    best_V = None
    best_W = None
    best_rank = rank_schoolbook
    best_err = np.inf

    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin

    # Try ranks from 119 down to 100
    for R in range(rank_schoolbook - 1, 99, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Scale to match magnitude
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Use a simple gradient descent or scipy minimize
            # Since we have limited time, a few steps of gradient descent might be enough to see if it's possible
            # Or use scipy.optimize.minimize
            
            def objective(params):
                U_flat = params[:nm * R].reshape(nm, R)
                V_flat = params[nm * R: nm * R + mp * R].reshape(mp, R)
                W_flat = params[nm * R + mp * R:].reshape(pn, R)
                
                # Compute tensor
                # T_est[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # This is a 3-way tensor product sum.
                # We can compute this efficiently using einsum
                T_est = np.einsum('ur,vr,wr->uvw', U_flat, V_flat, W_flat)
                err = np.sum((T - T_est) ** 2)
                return err

            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead or Powell? L-BFGS-B is better but might be slow.
            # Let's try a few steps of L-BFGS-B
            try:
                res = scipy.optimize.minimize(objective, params, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-5})
                if res.fun < best_err:
                    best_err = res.fun
                    best_rank = R
                    U_best = res.x[:nm * R].reshape(nm, R)
                    V_best = res.x[nm * R: nm * R + mp * R].reshape(mp, R)
                    W_best = res.x[nm * R + mp * R:].reshape(pn, R)
                    
                    # Check if error is small enough
                    if best_err < 1e-6:
                        # Found a valid decomposition
                        # Convert to integer/half-integer if possible? 
                        # The problem asks for integer/half-integer entries.
                        # If the solution is close to integers, we can round.
                        # But for now, return the float solution? 
                        # "with integer/half-integer entries that exactly decompose"
                        # If we find a float solution with error < 1e-6, it's likely an integer solution.
                        # We should try to round to nearest 0.5 or 1.
                        
                        U_rounded = np.round(U_best * 2) / 2
                        V_rounded = np.round(V_best * 2) / 2
                        W_rounded = np.round(W_best * 2) / 2
                        
                        # Verify rounded solution
                        T_est_rounded = np.einsum('ur,vr,wr->uvw', U_rounded, V_rounded, W_rounded)
                        err_rounded = np.sum((T - T_est_rounded) ** 2)
                        
                        if err_rounded < 1e-6:
                            # Convert to list of lists
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
            except Exception as e:
                pass

    # If no better rank found, return schoolbook
    U = [[0] * rank_schoolbook for _ in range(nm)]
    V = [[0] * rank_schoolbook for _ in range(mp)]
    W = [[0] * rank_schoolbook for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
