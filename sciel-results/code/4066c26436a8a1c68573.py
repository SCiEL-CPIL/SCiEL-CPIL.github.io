import scipy
import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Build target tensor T
    # T[I, J, K] where I in [0, n*m), J in [0, m*p), K in [0, p*n)
    # T[I, J, K] = 1 if consistent, else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U_sb[I, col] = 1
                V_sb[J, col] = 1
                W_sb[K, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Search for lower rank
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s limit
    
    def objective_and_grad(params, U_shape, V_shape, W_shape, T_target):
        nm, mp, pn = U_shape[0], V_shape[0], W_shape[0]
        R = U_shape[1]
        U_flat, V_flat, W_flat = np.split(params, [R*nm, R*nm + R*mp])
        U_opt = U_flat.reshape(nm, R)
        V_opt = V_flat.reshape(mp, R)
        W_opt = W_flat.reshape(pn, R)
        
        # Compute T_hat
        T_hat = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
        
        # Error
        err = T_hat - T_target
        obj = np.sum(err**2)
        
        # Gradient
        grad_U = 2.0 * np.einsum('ijk,jr,kr->ir', err, V_opt, W_opt)
        grad_V = 2.0 * np.einsum('ijk,ir,kr->jr', err, U_opt, W_opt)
        grad_W = 2.0 * np.einsum('ijk,ir,jr->kr', err, U_opt, V_opt)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return obj, grad
    
    def try_rank(R, max_restarts=10, time_per_restart=5.0):
        """Try to find a rank-R decomposition using L-BFGS-B with multiple restarts."""
        U_shape = (nm, R)
        V_shape = (mp, R)
        W_shape = (pn, R)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Small random
                scale = 0.1
            elif restart % 3 == 1:
                # Medium random
                scale = 0.5
            else:
                # Sparse-like random
                scale = 1.0
            
            U = np.random.randn(nm, R) * scale
            V = np.random.randn(mp, R) * scale
            W = np.random.randn(pn, R) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B with more iterations
            try:
                res = scipy.optimize.minimize(
                    lambda x: objective_and_grad(x, U_shape, V_shape, W_shape, T)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, U_shape, V_shape, W_shape, T)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
                )
            except Exception as e:
                continue
                
            if res.fun < 1e-6:
                # Extract optimized matrices
                U_opt, V_opt, W_opt = np.split(res.x, [R*nm, R*nm + R*mp])
                U_opt = U_opt.reshape(nm, R)
                V_opt = V_opt.reshape(mp, R)
                W_opt = W_opt.reshape(pn, R)
                
                # Try different rounding strategies
                for round_factor in [2, 4, 8]:
                    U_round = np.round(U_opt * round_factor) / round_factor
                    V_round = np.round(V_opt * round_factor) / round_factor
                    W_round = np.round(W_opt * round_factor) / round_factor
                    
                    # Verify
                    T_check = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                    if np.allclose(T_check, T, atol=1e-9):
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                
                # If rounding failed, try a polish step with alternating minimization
                U_polish = U_opt.copy()
                V_polish = V_opt.copy()
                W_polish = W_opt.copy()
                
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    # Update U
                    for i in range(nm):
                        # U[i,r] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r] / (sum_r' V[j,r']^2 * W[k,r']^2)
                        # Simplified: just do one step of gradient descent
                        pass
                    
                    # Actually, let's just try more L-BFGS-B iterations from this point
                    params2 = np.concatenate([U_polish.flatten(), V_polish.flatten(), W_polish.flatten()])
                    try:
                        res2 = scipy.optimize.minimize(
                            lambda x: objective_and_grad(x, U_shape, V_shape, W_shape, T)[0],
                            params2,
                            jac=lambda x: objective_and_grad(x, U_shape, V_shape, W_shape, T)[1],
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                    except Exception:
                        continue
                    
                    if res2.fun < 1e-8:
                        U_opt2, V_opt2, W_opt2 = np.split(res2.x, [R*nm, R*nm + R*mp])
                        U_opt2 = U_opt2.reshape(nm, R)
                        V_opt2 = V_opt2.reshape(mp, R)
                        W_opt2 = W_opt2.reshape(pn, R)
                        
                        for round_factor in [2, 4, 8, 16]:
                            U_round = np.round(U_opt2 * round_factor) / round_factor
                            V_round = np.round(V_opt2 * round_factor) / round_factor
                            W_round = np.round(W_opt2 * round_factor) / round_factor
                            
                            T_check = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                            if np.allclose(T_check, T, atol=1e-9):
                                return {
                                    "U": U_round.tolist(),
                                    "V": V_round.tolist(),
                                    "W": W_round.tolist()
                                }
        
        return None
    
    # Try ranks from 46 up to 55
    for R in range(46, 56):
        if time.time() > deadline:
            break
            
        result = try_rank(R, max_restarts=8, time_per_restart=5.0)
        if result is not None:
            best_result = result
            break

    return best_result
