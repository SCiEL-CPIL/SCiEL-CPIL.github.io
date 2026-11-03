import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
    
    # Schoolbook rank 27
    rank_sb = 27
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    best_rank = rank_sb
    
    def objective(vars, target_rank):
        r = target_rank
        U_f = vars[:nm*r].reshape(nm, r)
        V_f = vars[nm*r:2*nm*r].reshape(mp, r)
        W_f = vars[2*nm*r:3*nm*r].reshape(pn, r)
        
        errors = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W_f[k*n+i, :] * U_f[i*m+j, :] * V_f[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        errors.append(s - expected)
        return np.array(errors)
    
    def try_rounding(U_sol, V_sol, W_sol):
        """Try rounding to integer/half-integer/quarter values."""
        for scale in [1, 2, 4, 8]:
            U_r = np.round(U_sol * scale) / scale
            V_r = np.round(V_sol * scale) / scale
            W_r = np.round(W_sol * scale) / scale
            if verify(U_r, V_r, W_r, tol=1e-4):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        return None
    
    def polish_solution(U_init, V_init, W_init, target_rank, max_iter=300):
        """Polish a near-exact solution using high-precision least squares."""
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        try:
            res = least_squares(objective, x0, args=(target_rank,), method='lm', max_nfev=max_iter, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if res.cost < 1e-12:
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                return try_rounding(U_sol, V_sol, W_sol)
        except Exception:
            pass
        return None
    
    def try_reduce_rank(target_rank, num_starts=30):
        """Try to find a rank-`target_rank` decomposition."""
        rng = np.random.default_rng(42)
        
        for start_idx in range(num_starts):
            if time.time() > deadline - 3:
                return None
            
            if start_idx == 0:
                # Random initialization
                U_init = rng.standard_normal((nm, target_rank)) * 0.3
                V_init = rng.standard_normal((mp, target_rank)) * 0.3
                W_init = rng.standard_normal((pn, target_rank)) * 0.3
            elif start_idx == 1:
                # Structured initialization based on schoolbook
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                col_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_init[i*m+j, col_idx % target_rank] = 1.0
                            V_init[j*p+k, col_idx % target_rank] = 1.0
                            W_init[k*n+i, col_idx % target_rank] = 1.0
                            col_idx += 1
            elif start_idx == 2:
                # Strassen-inspired structure for 3x3
                # Use the known rank 19 or 20 approach if possible
                U_init = rng.standard_normal((nm, target_rank)) * 0.5
                V_init = rng.standard_normal((mp, target_rank)) * 0.5
                W_init = rng.standard_normal((pn, target_rank)) * 0.5
            else:
                # Perturbed random
                U_init = rng.standard_normal((nm, target_rank)) * 0.5
                V_init = rng.standard_normal((mp, target_rank)) * 0.5
                W_init = rng.standard_normal((pn, target_rank)) * 0.5
                # Add some structure
                U_init[i*m+j, start_idx % target_rank] += 1.0
                V_init[j*p+k, start_idx % target_rank] += 1.0
                W_init[k*n+i, start_idx % target_rank] += 1.0
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(objective, x0, args=(target_rank,), method='lm', max_nfev=150)
            except Exception:
                continue
                
            if res.cost < 1e-6:
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                
                polished = polish_solution(U_sol, V_sol, W_sol, target_rank)
                if polished:
                    return polished
                
                rounded = try_rounding(U_sol, V_sol, W_sol)
                if rounded:
                    return rounded
                
        return None
    
    # Try to find lower rank decompositions
    for target_rank in [19, 20, 21, 22, 23, 24, 25, 26]:
        if time.time() - start_time < deadline - 10:
            result = try_reduce_rank(target_rank, num_starts=20)
            if result:
                best_U, best_V, best_W = result
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
