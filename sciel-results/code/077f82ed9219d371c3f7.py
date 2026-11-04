import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def get_tensor_error(U, V, W, n, m, p, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Compute S[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # Using einsum: 'ijt,jkt,kit->ijk'
        S = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        # Error is sum of (S - 1)^2
        err = np.sum((S - 1.0)**2)
        return err, S

    def gradient(U, V, W, n, m, p, rank):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        S = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        dS = 2.0 * (S - 1.0) # (n, m, p)
        
        # dU[i,j,t] = sum_k dS[i,j,k] * V[j,k,t] * W[k,i,t]
        # dU = einsum('ijk,jkt,kit->ijt', dS, V3, W3)
        dU = np.einsum('ijk,jkt,kit->ijt', dS, V3, W3)
        
        # dV[j,k,t] = sum_i dS[i,j,k] * U[i,j,t] * W[k,i,t]
        # dV = einsum('ijk,ijt,kit->jkt', dS, U3, W3)
        dV = np.einsum('ijk,ijt,kit->jkt', dS, U3, W3)
        
        # dW[k,i,t] = sum_j dS[i,j,k] * U[i,j,t] * V[j,k,t]
        # dW = einsum('ijk,ijt,jkt->kit', dS, U3, V3)
        dW = np.einsum('ijk,ijt,jkt->kit', dS, U3, V3)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    def verify(U, V, W, n, m, p, rank, tol=1e-8):
        err, S = get_tensor_error(U, V, W, n, m, p, rank)
        return err < tol

    def try_rank(rank, seed):
        if time.time() > deadline - 10:
            return None
            
        rng = np.random.RandomState(seed)
        U0 = rng.randn(n * m, rank) * 0.5
        V0 = rng.randn(m * p, rank) * 0.5
        W0 = rng.randn(p * n, rank) * 0.5
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            err, _ = get_tensor_error(U, V, W, n, m, p, rank)
            return err
            
        def jac(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            return gradient(U, V, W, n, m, p, rank)
            
        try:
            res = minimize(objective, x0, jac=jac, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
        except:
            return None
            
        if res.success or res.fun < 1e-10:
            x_opt = res.x
            U = x_opt[:n*m*rank].reshape(n*m, rank)
            V = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Check if solution is close to half-integers
            U_dist = np.max(np.abs(U - np.round(U * 2) / 2))
            V_dist = np.max(np.abs(V - np.round(V * 2) / 2))
            W_dist = np.max(np.abs(W - np.round(W * 2) / 2))
            
            # Polish to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Try rounded solution first
            if verify(U_r, V_r, W_r, n, m, p, rank, tol=1e-6):
                return U_r, V_r, W_r
            
            # If rounding didn't work but original is very accurate and close to half-integers, 
            # try a higher precision polish
            if res.fun < 1e-12 and max(U_dist, V_dist, W_dist) < 0.01:
                # Do a final polish with tighter tolerance
                x_polish = x_opt.copy()
                def polish_objective(x):
                    U = x[:n*m*rank].reshape(n*m, rank)
                    V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    err, _ = get_tensor_error(U, V, W, n, m, p, rank)
                    return err
                
                def polish_jac(x):
                    U = x[:n*m*rank].reshape(n*m, rank)
                    V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    return gradient(U, V, W, n, m, p, rank)
                
                try:
                    res_polish = minimize(polish_objective, x_polish, jac=polish_jac, method='L-BFGS-B', 
                                          options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15})
                    if res_polish.success or res_polish.fun < 1e-14:
                        U_p = res_polish.x[:n*m*rank].reshape(n*m, rank)
                        V_p = res_polish.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                        W_p = res_polish.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        
                        U_r2 = np.round(U_p * 2) / 2
                        V_r2 = np.round(V_p * 2) / 2
                        W_r2 = np.round(W_p * 2) / 2
                        
                        if verify(U_r2, V_r2, W_r2, n, m, p, rank, tol=1e-5):
                            return U_r2, V_r2, W_r2
                except:
                    pass
        
        return None

    best_result = None
    for rank in [14, 15, 16, 17, 18, 19, 20]:
        if time.time() > deadline - 20:
            break
        for seed in range(15):
            if time.time() > deadline - 10:
                break
            res = try_rank(rank, seed)
            if res is not None:
                U, V, W = res
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                break
        if best_result:
            break

    if best_result:
        return best_result

    # Fallback schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
