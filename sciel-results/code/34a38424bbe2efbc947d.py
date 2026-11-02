import numpy as np
import time
from scipy.optimize import least_squares
import itertools

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 150s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
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

    def polish_solution(U_init, V_init, W_init, target_rank, max_iter=500):
        """Polish a near-exact solution using high-precision least squares."""
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        try:
            res = least_squares(objective, x0, args=(target_rank,), method='lm', max_nfev=max_iter, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if res.cost < 1e-12:
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                
                # Try rounding to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-4):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                # Try rounding to integers
                U_r = np.round(U_sol)
                V_r = np.round(V_sol)
                W_r = np.round(W_sol)
                
                if verify(U_r, V_r, W_r, tol=1e-4):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                # Try rounding to quarters
                U_r = np.round(U_sol * 4) / 4
                V_r = np.round(V_sol * 4) / 4
                W_r = np.round(W_sol * 4) / 4
                
                if verify(U_r, V_r, W_r, tol=1e-4):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    
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
                # Random initialization with structured pattern
                U_init = rng.standard_normal((nm, target_rank)) * 0.3
                V_init = rng.standard_normal((mp, target_rank)) * 0.3
                W_init = rng.standard_normal((pn, target_rank)) * 0.3
            elif start_idx == 1:
                # Structured initialization based on schoolbook
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                # Distribute the 27 schoolbook terms across target_rank columns
                terms_per_col = 27 // target_rank
                remainder = 27 % target_rank
                col_idx = 0
                term_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col_idx >= target_rank:
                                col_idx = 0
                            U_init[i*m+j, col_idx] = 1.0
                            V_init[j*p+k, col_idx] = 1.0
                            W_init[k*n+i, col_idx] = 1.0
                            col_idx += 1
                            if col_idx >= target_rank:
                                col_idx = 0
            else:
                # Perturbed schoolbook reduction
                keep_indices = rng.choice(rank_sb, target_rank, replace=False)
                sort_keep = np.sort(keep_indices)
                
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                for new_col, old_col in enumerate(sort_keep):
                    U_init[:, new_col] = U[:, old_col]
                    V_init[:, new_col] = V[:, old_col]
                    W_init[:, new_col] = W[:, old_col]
                
                # Add small perturbation to escape local minima
                U_init += rng.standard_normal((nm, target_rank)) * 0.1
                V_init += rng.standard_normal((mp, target_rank)) * 0.1
                W_init += rng.standard_normal((pn, target_rank)) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(objective, x0, args=(target_rank,), method='lm', max_nfev=200)
            except Exception:
                continue
                
            if res.cost < 1e-6:
                # Found a good solution, polish it
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                
                polished = polish_solution(U_sol, V_sol, W_sol, target_rank)
                if polished:
                    return polished
                
                # If polish failed, try rounding directly
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-3):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                U_r = np.round(U_sol)
                V_r = np.round(V_sol)
                W_r = np.round(W_sol)
                
                if verify(U_r, V_r, W_r, tol=1e-3):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
        return None

    # Try rank 23 first (known to be achievable for 3x3x3)
    if time.time() - start_time < deadline - 15:
        result = try_reduce_rank(23, num_starts=25)
        if result:
            best_U, best_V, best_W = result
            best_rank = 23
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 25
    if time.time() - start_time < deadline - 15:
        result = try_reduce_rank(25, num_starts=15)
        if result:
            best_U, best_V, best_W = result
            best_rank = 25
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 26
    if time.time() - start_time < deadline - 10:
        result = try_reduce_rank(26, num_starts=10)
        if result:
            best_U, best_V, best_W = result
            best_rank = 26
            return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
