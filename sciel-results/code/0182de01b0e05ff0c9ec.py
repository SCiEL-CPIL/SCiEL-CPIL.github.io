import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    
    # Precompute indices for U, V, W for each (i, j, k)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    U_idx = []
    V_idx = []
    W_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_idx.append(i * m + j)
                V_idx.append(j * p + k)
                W_idx.append(k * n + i)
                
    U_idx = np.array(U_idx)
    V_idx = np.array(V_idx)
    W_idx = np.array(W_idx)
    
    num_entries = n * m * p
    
    def try_rank(rank, max_time=100):
        deadline = start_time + max_time
        if time.time() > deadline:
            return None
            
        # Try a few random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 1000 + rank)
            
            # Initialize
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            def obj_and_grad(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                # Compute values
                U_sel = U[U_idx]  # (84, rank)
                V_sel = V[V_idx]  # (84, rank)
                W_sel = W[W_idx]  # (84, rank)
                
                # Product and sum
                prod = U_sel * V_sel * W_sel
                vals = np.sum(prod, axis=1)  # (84,)
                
                # Error
                diff = vals - 1.0
                error = np.sum(diff**2)
                
                # Gradient
                dvals = 2 * diff  # (84,)
                
                # dU
                dU_contrib = dvals[:, None] * V_sel * W_sel  # (84, rank)
                dU = np.zeros_like(U)
                np.add.at(dU, U_idx, dU_contrib)
                
                # dV
                dV_contrib = dvals[:, None] * U_sel * W_sel  # (84, rank)
                dV = np.zeros_like(V)
                np.add.at(dV, V_idx, dV_contrib)
                
                # dW
                dW_contrib = dvals[:, None] * U_sel * V_sel  # (84, rank)
                dW = np.zeros_like(W)
                np.add.at(dW, W_idx, dW_contrib)
                
                grad = np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
                
                return error, grad
            
            try:
                res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            except:
                continue
                
            if time.time() > deadline:
                break
                
            U_opt = res.x[:n*m*rank].reshape(n*m, rank)
            V_opt = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
            W_opt = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest 0.5
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            # Verify
            U_sel = U_round[U_idx]
            V_sel = V_round[V_idx]
            W_sel = W_round[W_idx]
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            
            if np.allclose(vals, 1.0, atol=1e-9):
                return U_round.tolist(), V_round.tolist(), W_round.tolist()
                
        return None

    # Try ranks from low to high
    # Lower bound is around 30-40 for these dimensions usually
    for rank in range(30, 85):
        if time.time() > start_time + 110:
            break
        result = try_rank(rank, max_time=110 - (time.time() - start_time))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
