"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Precompute valid triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    
    ta = np.array([t[0] for t in triples], dtype=np.int32)
    tb = np.array([t[1] for t in triples], dtype=np.int32)
    tc = np.array([t[2] for t in triples], dtype=np.int32)
    
    def obj_and_grad(x, r):
        """Objective and gradient for rank r."""
        U = x[:8*r].reshape(8, r)
        V = x[8*r:8*r+28*r].reshape(28, r)
        W = x[8*r+28*r:].reshape(14, r)
        
        # vals[t] = sum_r U[ta[t],r] * V[tb[t],r] * W[tc[t],r]
        Us = U[ta, :]  # (56, r)
        Vs = V[tb, :]  # (56, r)
        Ws = W[tc, :]  # (56, r)
        
        prod = Us * Vs * Ws  # (56, r)
        vals = prod.sum(axis=1)  # (56,)
        obj = ((vals - 1.0) ** 2).sum()
        
        # Gradient
        dvals = 2.0 * (vals - 1.0)  # (56,)
        dprod = dvals[:, None] * (Vs * Ws)  # (56, r) -- d(obj)/d(Us)
        dUs = dprod  # (56, r)
        dVs = dvals[:, None] * (Us * Ws)  # (56, r)
        dWs = dvals[:, None] * (Us * Vs)  # (56, r)
        
        # Scatter back
        dU = np.zeros((8, r))
        dV = np.zeros((28, r))
        dW = np.zeros((14, r))
        np.add.at(dU, ta, dUs)
        np.add.at(dV, tb, dVs)
        np.add.at(dW, tc, dWs)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return obj, grad
    
    def try_rank(r, n_trials=3, max_iter=50):
        best_err = float('inf')
        best_x = None
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(7000 + r * 100 + trial)
            scale = (1.0 / r) ** (1.0/3.0)
            x0 = np.random.randn(50 * r) * scale
            
            from scipy.optimize import minimize
            try:
                res = minimize(obj_and_grad, x0, args=(r,), method='L-BFGS-B',
                              jac=True, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 20})
                if res.fun < best_err:
                    best_err = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        if best_x is not None and best_err < 1e-8:
            U = best_x[:8*r].reshape(8, r)
            V = best_x[8*r:8*r+28*r].reshape(28, r)
            W = best_x[8*r+28*r:].reshape(14, r)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            Us = U_r[ta, :]
            Vs = V_r[tb, :]
            Ws = W_r[tc, :]
            vals = (Us * Vs * Ws).sum(axis=1)
            if np.max(np.abs(vals - 1.0)) < 1e-6:
                return U_r, V_r, W_r
        
        return None
    
    # Try ranks
    for r in [48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35]:
        if time.time() > deadline - 10:
            break
        result = try_rank(r, n_trials=2, max_iter=40)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
