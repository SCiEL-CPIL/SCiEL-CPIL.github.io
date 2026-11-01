import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 5
    
    def get_rank_18_solution():
        rank = 18
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_strassen_block(t_offset, k_start):
            U[0, t_offset+0] = 1
            U[3, t_offset+0] = 1
            U[2, t_offset+1] = 1
            U[3, t_offset+1] = 1
            U[0, t_offset+2] = 1
            U[3, t_offset+3] = 1
            U[0, t_offset+4] = 1
            U[1, t_offset+4] = 1
            U[2, t_offset+5] = 1
            U[0, t_offset+5] = -1
            U[1, t_offset+6] = 1
            U[3, t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0, t_offset+0] = 1
            V[1*p + k1, t_offset+0] = 1
            V[0*p + k0, t_offset+1] = 1
            V[0*p + k1, t_offset+2] = 1
            V[1*p + k1, t_offset+2] = -1
            V[1*p + k0, t_offset+3] = 1
            V[0*p + k0, t_offset+3] = -1
            V[1*p + k1, t_offset+4] = 1
            V[0*p + k0, t_offset+5] = 1
            V[0*p + k1, t_offset+5] = 1
            V[1*p + k0, t_offset+6] = 1
            V[1*p + k1, t_offset+6] = 1
            
            W[k0*n + 0, t_offset+0] = 1
            W[k1*n + 1, t_offset+0] = 1
            W[k0*n + 1, t_offset+1] = 1
            W[k1*n + 1, t_offset+1] = -1
            W[k1*n + 0, t_offset+2] = 1
            W[k1*n + 1, t_offset+2] = 1
            W[k0*n + 0, t_offset+3] = 1
            W[k0*n + 1, t_offset+3] = 1
            W[k0*n + 0, t_offset+4] = -1
            W[k1*n + 0, t_offset+4] = 1
            W[k1*n + 1, t_offset+5] = 1
            W[k0*n + 0, t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0, t] = 1
        V[0*p + 4, t] = 1
        W[4*n + 0, t] = 1
        t += 1
        
        U[1, t] = 1
        V[1*p + 4, t] = 1
        W[4*n + 0, t] = 1
        t += 1
        
        U[2, t] = 1
        V[0*p + 4, t] = 1
        W[4*n + 1, t] = 1
        t += 1
        
        U[3, t] = 1
        V[1*p + 4, t] = 1
        W[4*n + 1, t] = 1
        
        return U, V, W

    def residual_full(params, rank):
        U = params[:n*m*rank].reshape((n*m, rank))
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
        W = params[n*m*rank + m*p*rank :].reshape((p*n, rank))
        
        res = np.zeros(n * m * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        target = 1.0 if j == k else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[k*p + l, c] * W[l*n + i, c]
                        res[idx] = val - target
                        idx += 1
        return res

    def check_exact_full(U, V, W, rank, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        target = 1.0 if j == k else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[k*p + l, c] * W[l*n + i, c]
                        if abs(val - target) > tol:
                            return False
        return True

    def snap_to_half_integers(U, V, W, rank):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    def try_rank_from_perturbation(target_rank, base_U, base_V, base_W, base_rank, num_restarts=40, pert_scale=0.3):
        if time.time() > DEADLINE - 15:
            return None
        
        size = n*m*target_rank + m*p*target_rank + p*n*target_rank
        best_err = float('inf')
        best_x = None
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            np.random.seed(restart * 97 + target_rank * 13)
            
            # Create perturbed version of base solution, truncated to target_rank
            # Take first target_rank columns from base and perturb
            if target_rank <= base_rank:
                U_base = base_U[:, :target_rank] + np.random.normal(0, pert_scale, (n*m, target_rank))
                V_base = base_V[:, :target_rank] + np.random.normal(0, pert_scale, (m*p, target_rank))
                W_base = base_W[:, :target_rank] + np.random.normal(0, pert_scale, (p*n, target_rank))
            else:
                U_base = np.zeros((n*m, target_rank))
                V_base = np.zeros((m*p, target_rank))
                W_base = np.zeros((p*n, target_rank))
                if base_rank > 0:
                    U_base[:, :base_rank] = base_U + np.random.normal(0, pert_scale, (n*m, base_rank))
                    V_base[:, :base_rank] = base_V + np.random.normal(0, pert_scale, (m*p, base_rank))
                    W_base[:, :base_rank] = base_W + np.random.normal(0, pert_scale, (p*n, base_rank))
            
            x0 = np.concatenate([U_base.flatten(), V_base.flatten(), W_base.flatten()])
            
            try:
                res = least_squares(residual_full, x0, args=(target_rank,), method='lm', 
                                    max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                err = float(np.sum(res.fun**2))
                if err < best_err:
                    best_err = err
                    best_x = res.x.copy()
                if err < 1e-8:
                    break
            except Exception:
                continue
        
        if best_x is not None and best_err < 1e-5:
            U = best_x[:n*m*target_rank].reshape((n*m, target_rank))
            V = best_x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
            W = best_x[n*m*target_rank + m*p*target_rank :].reshape((p*n, target_rank))
            
            if check_exact_full(U, V, W, target_rank):
                U_r, V_r, W_r = snap_to_half_integers(U, V, W, target_rank)
                if check_exact_full(U_r, V_r, W_r, target_rank):
                    return U_r, V_r, W_r
                else:
                    return U, V, W
        return None

    def try_rank_random(target_rank, num_restarts=30):
        if time.time() > DEADLINE - 15:
            return None
        
        size = n*m*target_rank + m*p*target_rank + p*n*target_rank
        best_err = float('inf')
        best_x = None
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            np.random.seed(restart * 101 + target_rank * 7)
            x0 = np.random.uniform(-1, 1, size=size)
            
            try:
                res = least_squares(residual_full, x0, args=(target_rank,), method='lm', 
                                    max_nfev=1500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                err = float(np.sum(res.fun**2))
                if err < best_err:
                    best_err = err
                    best_x = res.x.copy()
                if err < 1e-8:
                    break
            except Exception:
                continue
        
        if best_x is not None and best_err < 1e-5:
            U = best_x[:n*m*target_rank].reshape((n*m, target_rank))
            V = best_x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
            W = best_x[n*m*target_rank + m*p*target_rank :].reshape((p*n, target_rank))
            
            if check_exact_full(U, V, W, target_rank):
                U_r, V_r, W_r = snap_to_half_integers(U, V, W, target_rank)
                if check_exact_full(U_r, V_r, W_r, target_rank):
                    return U_r, V_r, W_r
                else:
                    return U, V, W
        return None

    # Get base rank-18 solution
    U18, V18, W18 = get_rank_18_solution()
    
    # Try rank 17 with perturbation from rank-18
    if time.time() < DEADLINE - 25:
        sol = try_rank_from_perturbation(17, U18, V18, W18, 18, num_restarts=50, pert_scale=0.5)
        if sol:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 17 with random init
    if time.time() < DEADLINE - 20:
        sol = try_rank_random(17, num_restarts=30)
        if sol:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 16 with perturbation from rank-18
    if time.time() < DEADLINE - 20:
        sol = try_rank_from_perturbation(16, U18, V18, W18, 18, num_restarts=40, pert_scale=0.7)
        if sol:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 16 with random init
    if time.time() < DEADLINE - 15:
        sol = try_rank_random(16, num_restarts=20)
        if sol:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 18
    return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
