"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import random
import sys
import numpy as np
from scipy.optimize import minimize

DEADLINE = 140  # safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24
    
    # Precompute index mappings for tensor evaluation
    # For each (i,j,k), we need U index = i*m+j, V index = j*p+k, W index = k*n+i
    idx_U = np.array([i*m+j for i in range(n) for j in range(m) for k in range(p)])
    idx_V = np.array([j*p+k for i in range(n) for j in range(m) for k in range(p)])
    idx_W = np.array([k*n+i for i in range(n) for j in range(m) for k in range(p)])
    n_entries = n * m * p  # 120
    
    # Schoolbook: rank 120
    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def tensor_eval(U, V, W):
        """Compute all 120 tensor entries efficiently."""
        # U shape (nm, R), V shape (mp, R), W shape (pn, R)
        # For each of the 120 (i,j,k) triples, compute sum_r U[idx_U[t],r]*V[idx_V[t],r]*W[idx_W[t],r]
        U_sel = U[idx_U, :]  # (120, R)
        V_sel = V[idx_V, :]  # (120, R)
        W_sel = W[idx_W, :]  # (120, R)
        return np.sum(U_sel * V_sel * W_sel, axis=1)  # (120,)
    
    def try_rank(R, time_budget):
        """Try to find exact rank-R decomposition via optimization."""
        deadline = time.time() + time_budget
        
        best_sol = None
        best_obj = float('inf')
        
        n_restarts = 0
        max_restarts = 50
        while time.time() < deadline - 3 and n_restarts < max_restarts:
            n_restarts += 1
            
            # Initialize
            if n_restarts == 1:
                # First try: structured init close to schoolbook pattern
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                # Assign schoolbook-like structure with perturbation
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = (i*m+j)*p+k  # this exceeds R if R < 120, so mod
                            r = r % R
                            U[i*m+j, r] = 1.0
                            V[j*p+k, r] = 1.0
                            W[k*n+i, r] = 1.0
                # Add small random perturbation
                U += np.random.randn(nm, R) * 0.1
                V += np.random.randn(mp, R) * 0.1
                W += np.random.randn(pn, R) * 0.1
            else:
                # Random init with structure
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
                for c in range(R):
                    for _ in range(3):
                        i = random.randint(0, n - 1)
                        j = random.randint(0, m - 1)
                        k = random.randint(0, p - 1)
                        U[i * m + j, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                        V[j * p + k, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                        W[k * n + i, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
            
            def obj_grad(x):
                Ux = x[:nm*R].reshape(nm, R)
                Vx = x[nm*R:nm*R+mp*R].reshape(mp, R)
                Wx = x[nm*R+mp*R:][:, :R].reshape(pn, R)
                
                # Compute tensor values
                T = tensor_eval(Ux, Vx, Wx)  # (120,)
                diff = T - 1.0  # (120,)
                fval = float(np.sum(diff * diff))
                
                # Gradients
                # dU[i*m+j, r] = 2 * sum over t where idx_U[t]==i*m+j of diff[t] * Vx[idx_V[t],r] * Wx[idx_W[t],r]
                dU = np.zeros((nm, R))
                dV = np.zeros((mp, R))
                dW = np.zeros((pn, R))
                
                # Vectorized gradient computation
                U_sel = Ux[idx_U, :]  # (120, R)
                V_sel = Vx[idx_V, :]  # (120, R)
                W_sel = Wx[idx_W, :]  # (120, R)
                
                # diff is (120,), need to scatter gradients
                # dU[i*m+j, r] += 2 * diff[t] * V_sel[t, r] * W_sel[t, r] for t where idx_U[t] == i*m+j
                # Use np.add.at for unbuffered accumulation
                dU_contrib = 2.0 * diff[:, None] * V_sel * W_sel  # (120, R)
                dV_contrib = 2.0 * diff[:, None] * U_sel * W_sel  # (120, R)
                dW_contrib = 2.0 * diff[:, None] * U_sel * V_sel  # (120, R)
                
                np.add.at(dU, idx_U, dU_contrib)
                np.add.at(dV, idx_V, dV_contrib)
                np.add.at(dW, idx_W, dW_contrib)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return fval, grad
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B with the combined objective+gradient
                res = minimize(lambda x: obj_grad(x)[0], x0, 
                              jac=lambda x: obj_grad(x)[1],
                              method='L-BFGS-B',
                              options={'maxiter': 300, 'maxfun': 10000,
                                       'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100})
                fval = res.fun
                if fval < best_obj:
                    best_obj = fval
                    x_opt = res.x
                    U_opt = x_opt[:nm*R].reshape(nm, R)
                    V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = x_opt[nm*R+mp*R:][:, :R].reshape(pn, R)
                    best_sol = (U_opt, V_opt, W_opt)
            except Exception:
                pass
            
            if best_obj < 1e-14:
                break
        
        # Final polish: if we have a near-exact solution, try to refine
        if best_sol is not None and best_obj < 1e-8:
            U_opt, V_opt, W_opt = best_sol
            x0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
            try:
                res = minimize(lambda x: obj_grad(x)[0], x0,
                              jac=lambda x: obj_grad(x)[1],
                              method='L-BFGS-B',
                              options={'maxiter': 500, 'maxfun': 20000,
                                       'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 200})
                if res.fun < best_obj:
                    best_obj = res.fun
                    x_opt = res.x
                    U_opt = x_opt[:nm*R].reshape(nm, R)
                    V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = x_opt[nm*R+mp*R:][:, :R].reshape(pn, R)
                    best_sol = (U_opt, V_opt, W_opt)
            except Exception:
                pass
        
        # Verify exactness
        if best_sol is not None and best_obj < 1e-10:
            U_opt, V_opt, W_opt = best_sol
            T = tensor_eval(U_opt, V_opt, W_opt)
            max_viol = float(np.max(np.abs(T - 1.0)))
            
            if max_viol < 1e-10:
                U_list = [[float(U_opt[i, r]) for r in range(R)] for i in range(nm)]
                V_list = [[float(V_opt[i, r]) for r in range(R)] for i in range(mp)]
                W_list = [[float(W_opt[i, r]) for r in range(R)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    ranks_to_try = [48, 54, 60, 64, 72, 80, 90, 100, 110, 119]
    
    for R in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        time_left = DEADLINE - time.time() - 5
        if time_left < 5:
            break
        
        result = try_rank(R, min(time_left, 40))
        if result is not None:
            best = result
            break
    
    return best
