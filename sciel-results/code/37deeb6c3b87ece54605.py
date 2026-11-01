"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Naive rank 16 decomposition
    def naive_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def check_decomposition(U, V, W, R):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    # Wait, the term is sum_r U[i,j,r] V[j,k,r] W[i,k,r]
                    # So it's element-wise product of V and W rows, then dot with U row?
                    # No, sum_r U[r] V[r] W[r].
                    # So: np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def loss_grad(params, R):
        # Unpack
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Term: sum_r U[i,j,r] V[j,k,r] W[i,k,r] - 1
                    res = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
                    loss += res * res
                    
                    # Gradients
                    # d/dU[i,j,r] = 2 * res * V[j,k,r] * W[i,k,r]
                    # d/dV[j,k,r] = 2 * res * U[i,j,r] * W[i,k,r]
                    # d/dW[i,k,r] = 2 * res * U[i,j,r] * V[j,k,r]
                    
                    grad_U[i*m+j, :] += 2 * res * V[j*p+k, :] * W[k*n+i, :]
                    grad_V[j*p+k, :] += 2 * res * U[i*m+j, :] * W[k*n+i, :]
                    grad_W[k*n+i, :] += 2 * res * U[i*m+j, :] * V[j*p+k, :]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_rank(R, max_restarts=50):
        if time.time() > deadline:
            return None
            
        np.random.seed(42)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            # Random initialization
            # Scale: values around 1.0
            params0 = np.random.uniform(-1, 1, nm*R + mp*R + pn*R)
            
            res = minimize(
                lambda x: loss_grad(x, R)[0], 
                params0, 
                jac=lambda x: loss_grad(x, R)[1], 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success and res.fun < 1e-8:
                # Extract U, V, W
                U = res.x[:nm*R].reshape(nm, R)
                V = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W = res.x[nm*R+mp*R:].reshape(pn, R)
                
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if check_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), R):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
                # Try integer snap
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if check_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), R):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
            # If loss is very low but snap failed, maybe try a local search around the solution?
            # Or just accept if loss is extremely low and we can't snap, but the task requires integer/half-integer.
            # So we must snap.
            
        return None

    # Try ranks from 10 upwards
    for R in range(10, 16):
        result = try_rank(R)
        if result is not None:
            return result
            
    # Fallback to naive
    return naive_decomp()
# EVOLVE-BLOCK-END
