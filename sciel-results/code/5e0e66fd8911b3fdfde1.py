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
    deadline = start_time + 110 # 10s margin
    
    # Try ranks from 46 up to 55
    # We know 46 is possible.
    for R in range(46, 56):
        if time.time() > deadline:
            break
            
        success = False
        # Multiple restarts for each rank
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use small random values
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            # Optimize
            # We will use a simple gradient descent with adaptive step size
            # or scipy L-BFGS-B
            
            def objective_and_grad(params):
                # params is flat array of size R*(nm+mp+pn)
                U_flat, V_flat, W_flat = np.split(params, [R*nm, R*nm + R*mp])
                U_opt = U_flat.reshape(nm, R)
                V_opt = V_flat.reshape(mp, R)
                W_opt = W_flat.reshape(pn, R)
                
                # Compute T_hat
                # T_hat[I, J, K] = sum_r U[I,r] V[J,r] W[K,r]
                # This is a tensor contraction.
                # Efficiently:
                # T_hat = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
                T_hat = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
                
                # Error
                err = T_hat - T
                obj = np.sum(err**2)
                
                # Gradient
                # d(obj)/dU[I,r] = 2 * sum_{J,K} err[I,J,K] V[J,r] W[K,r]
                # d(obj)/dV[J,r] = 2 * sum_{I,K} err[I,J,K] U[I,r] W[K,r]
                # d(obj)/dW[K,r] = 2 * sum_{I,J} err[I,J,K] U[I,r] V[J,r]
                
                # grad_U[I, r] = 2 * einsum('ijk,jr,kr->ir', err, V_opt, W_opt)
                grad_U = 2.0 * np.einsum('ijk,jr,kr->ir', err, V_opt, W_opt)
                grad_V = 2.0 * np.einsum('ijk,ir,kr->jr', err, U_opt, W_opt)
                grad_W = 2.0 * np.einsum('ijk,ir,jr->kr', err, U_opt, V_opt)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                return obj, grad
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B
            try:
                res = scipy.optimize.minimize(
                    lambda x: objective_and_grad(x)[0],
                    params,
                    jac=lambda x: objective_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
                )
            except Exception as e:
                continue
                
            if res.success or res.fun < 1e-4:
                # Check exactness
                U_opt, V_opt, W_opt = np.split(res.x, [R*nm, R*nm + R*mp])
                U_opt = U_opt.reshape(nm, R)
                V_opt = V_opt.reshape(mp, R)
                W_opt = W_opt.reshape(pn, R)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify
                T_check = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                if np.allclose(T_check, T, atol=1e-9):
                    # Success
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    success = True
                    break
            else:
                # Maybe try a few more iterations or different init?
                # For now, just break to next restart
                pass
        
        if success:
            break

    return best_result
