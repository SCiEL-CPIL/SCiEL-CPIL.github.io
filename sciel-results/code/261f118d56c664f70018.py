import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Try ranks lower than schoolbook
    for R in range(13, rank_school):
        if time.time() > deadline - 10:
            break
            
        # Try multiple initializations
        for init in range(20):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale: entries should be around 1/sqrt(R)
            U = np.random.randn(n * m, R) / np.sqrt(R)
            V = np.random.randn(m * p, R) / np.sqrt(R)
            W = np.random.randn(p * n, R) / np.sqrt(R)
            
            # Optimize
            # Define objective and gradient
            def obj_grad(params):
                # Reshape params
                U_flat = params[:n*m*R].reshape(n*m, R)
                V_flat = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = params[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute S
                S = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                # Residual
                res = S - 1.0
                loss = np.sum(res**2)
                
                # Gradient
                dU = np.zeros_like(U_flat)
                dV = np.zeros_like(V_flat)
                dW = np.zeros_like(W_flat)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r_val = res[i,j,k]
                            # dU[i*m+j, :] += 2 * r_val * V[j*p+k, :] * W[k*n+i, :]
                            dU[i*m+j, :] += 2 * r_val * V_flat[j*p+k, :] * W_flat[k*n+i, :]
                            dV[j*p+k, :] += 2 * r_val * U_flat[i*m+j, :] * W_flat[k*n+i, :]
                            dW[k*n+i, :] += 2 * r_val * U_flat[i*m+j, :] * V_flat[j*p+k, :]
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad

            # Use scipy.optimize.minimize
            # We need to pass the gradient
            from scipy.optimize import minimize
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # We want to check if we can find a solution
            # Since we don't know the scale, we might need to adjust.
            # But BFGS should handle it.
            
            try:
                res_opt = minimize(
                    lambda x: obj_grad(x)[0],
                    x0,
                    jac=lambda x: obj_grad(x)[1],
                    method='BFGS',
                    options={'maxiter': 50, 'disp': False}
                )
            except Exception:
                continue

            if res_opt.success or res_opt.fun < 1e-6:
                # Extract U, V, W
                U_opt = res_opt.x[:n*m*R].reshape(n*m, R)
                V_opt = res_opt.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res_opt.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Check if close to half-integers
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify
                S_check = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S_check[i,j,k] = np.sum(U_snap[i*m+j, :] * V_snap[j*p+k, :] * W_snap[k*n+i, :])
                
                if np.allclose(S_check, 1.0, atol=1e-9):
                    # Valid decomposition
                    best_rank = R
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    # We found a lower rank, break inner loop
                    break
            # If not successful, continue to next init

        # If we found a decomposition for this R, we can try lower R?
        # The loop goes from 13 upwards. So if we find R=13, we break the outer loop?
        # No, we want the MINIMAL rank.
        # So if we find R=13, that's the best possible (likely).
        # If we don't find R=13, we try R=14, etc.
        # So the loop should be from 13 to 17.
        # If we find a valid one at R, we should check if we can find lower?
        # No, the loop is increasing. So the first one found is the lowest in the range.
        # But we should break if we find one?
        # Yes, because we are iterating from low to high.
        break

    return best_result
