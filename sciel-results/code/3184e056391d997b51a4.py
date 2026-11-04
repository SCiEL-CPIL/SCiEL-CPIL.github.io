import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify(U, V, W):
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
            if np.any(np.abs(mat) > 100):
                return False
        
        try:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if not np.isclose(val, 1.0, atol=1e-6):
                            return False
        except Exception:
            return False
        return True

    def loss_and_grad(params, R):
        # params shape: (3, R, dim) flattened
        U = params[:R*n*m].reshape(R, n*m)
        V = params[R*n*m:2*R*n*m].reshape(R, m*p)
        W = params[2*R*n*m:3*R*n*m].reshape(R, p*n)
        
        # Compute loss: sum of (sum_r U[r,i,j]*V[r,j,k]*W[r,k,i] - 1)^2
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # val = sum_r U[r,i*m+j] * V[r,j*p+k] * W[r,k*n+i]
                    val = np.sum(U[:, i*m+j] * V[:, j*p+k] * W[:, k*n+i])
                    err = val - 1.0
                    loss += err * err
                    
                    # d(loss)/d(U[r,i,j]) = 2*err * V[r,j,k] * W[r,k,i]
                    grad_U[:, i*m+j] += 2 * err * V[:, j*p+k] * W[:, k*n+i]
                    grad_V[:, j*p+k] += 2 * err * U[:, i*m+j] * W[:, k*n+i]
                    grad_W[:, k*n+i] += 2 * err * U[:, i*m+j] * V[:, j*p+k]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def solve_rank(R, max_restarts=10):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Structured initialization
            U = np.random.randn(R, n*m) * 0.1
            V = np.random.randn(R, m*p) * 0.1
            W = np.random.randn(R, p*n) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B for optimization
            result = minimize(
                lambda x: loss_and_grad(x, R)[0],
                params,
                jac=lambda x: loss_and_grad(x, R)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                return None
            
            U_opt = result.x[:R*n*m].reshape(R, n*m).T
            V_opt = result.x[R*n*m:2*R*n*m].reshape(R, m*p).T
            W_opt = result.x[2*R*n*m:3*R*n*m].reshape(R, p*n).T
            
            # Round to half-integers
            U_int = np.round(U_opt * 2) / 2.0
            V_int = np.round(V_opt * 2) / 2.0
            W_int = np.round(W_opt * 2) / 2.0
            
            if verify(U_int, V_int, W_int):
                return U_int, V_int, W_int
        
        return None

    for R in range(19, 25):
        if time.time() > deadline:
            break
        
        result = solve_rank(R)
        if result:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
