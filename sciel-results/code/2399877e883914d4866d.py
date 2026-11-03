import time
import numpy as np
import random

# Precompute indices for the 3x3x7 tensor
n, m, p = 3, 3, 7
num_I = n * m  # 9
num_J = m * p  # 21
num_K = p * n  # 21

# On-positions: T[I,J,K] = 1
on_I = []
on_J = []
on_K = []
for i in range(n):
    for j in range(m):
        for k in range(p):
            on_I.append(i * m + j)
            on_J.append(j * p + k)
            on_K.append(k * n + i)

on_I = np.array(on_I)
on_J = np.array(on_J)
on_K = np.array(on_K)
num_on = len(on_I)  # 63

# Off-positions: T[I,J,K] = 0 (all other positions)
all_I = np.arange(num_I * num_J * num_K)
all_I_flat = np.repeat(np.arange(num_I), num_J * num_K)
all_J_flat = np.tile(np.repeat(np.arange(num_J), num_K), num_I)
all_K_flat = np.tile(np.arange(num_K), num_I * num_J)

# Create mask for on-positions
on_set = set(zip(on_I.tolist(), on_J.tolist(), on_K.tolist()))
off_mask = np.array([
    (all_I_flat[s], all_J_flat[s], all_K_flat[s]) not in on_set
    for s in range(len(all_I_flat))
])
off_I = all_I_flat[off_mask]
off_J = all_J_flat[off_mask]
off_K = all_K_flat[off_mask]
num_off = len(off_I)  # 3906

def compute_error(params, r):
    """
    Compute the sum of squared errors for the full tensor decomposition.
    params: flattened vector of U, V, W.
    r: rank.
    """
    U = params[0 : num_I*r].reshape(num_I, r)
    V = params[num_I*r : num_I*r + num_J*r].reshape(num_J, r)
    W = params[num_I*r + num_J*r : num_I*r + num_J*r + num_K*r].reshape(num_K, r)
    
    # Compute contributions for on-positions
    T_on = np.zeros(num_on)
    for t in range(r):
        T_on += U[on_I, t] * V[on_J, t] * W[on_K, t]
    
    # Compute contributions for off-positions
    T_off = np.zeros(num_off)
    for t in range(r):
        T_off += U[off_I, t] * V[off_J, t] * W[off_K, t]
    
    # Error: on-positions should be 1, off-positions should be 0
    err_on = np.sum((T_on - 1.0) ** 2)
    err_off = np.sum(T_off ** 2)
    
    return err_on + err_off

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin below 150s kill
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((num_I, rank_schoolbook))
    V_school = np.zeros((num_J, rank_schoolbook))
    W_school = np.zeros((num_K, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_solution = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find lower rank solutions
    # Based on tensor rank theory, for 3x3x7, rank should be significantly less than 63
    # but more than 23 (the 3x3x3 case). Try a range of candidates.
    candidate_ranks = [55, 50, 45, 40, 35, 30, 28, 25]
    
    from scipy.optimize import minimize
    
    for r in candidate_ranks:
        if time.time() - start_time > time_limit - 15:
            break
            
        found_valid = False
        # Try multiple initialization strategies
        init_strategies = [
            ('random_small', lambda: (
                np.random.rand(num_I, r) * 0.5 + 0.25,
                np.random.rand(num_J, r) * 0.5 + 0.25,
                np.random.rand(num_K, r) * 0.5 + 0.25
            )),
            ('random_large', lambda: (
                np.random.rand(num_I, r) * 1.0,
                np.random.rand(num_J, r) * 1.0,
                np.random.rand(num_K, r) * 1.0
            )),
            ('structured', lambda: (
                np.ones((num_I, r)) * 0.5,
                np.ones((num_J, r)) * 0.5,
                np.ones((num_K, r)) * 0.5
            )),
        ]
        
        for trial in range(3):
            if time.time() - start_time > time_limit - 10:
                break
                
            random.seed(r * 100 + trial * 7 + 42)
            np.random.seed(r * 100 + trial * 7 + 42)
            
            # Cycle through initialization strategies
            strat_name, strat_fn = init_strategies[trial % len(init_strategies)]
            U0, V0, W0 = strat_fn()
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Optimize
            try:
                def objective(params):
                    return compute_error(params, r)
                    
                res = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    params_opt = res.x
                    U_opt = params_opt[0 : num_I*r].reshape(num_I, r)
                    V_opt = params_opt[num_I*r : num_I*r + num_J*r].reshape(num_J, r)
                    W_opt = params_opt[num_I*r + num_J*r : num_I*r + num_J*r + num_K*r].reshape(num_K, r)
                    
                    # Check error
                    err = compute_error(params_opt, r)
                    if err < 1e-6:
                        # Try to snap to integers/half-integers
                        U_snap = np.round(U_opt * 2) / 2
                        V_snap = np.round(V_opt * 2) / 2
                        W_snap = np.round(W_opt * 2) / 2
                        
                        # Verify snapped solution
                        params_snap = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                        err_snap = compute_error(params_snap, r)
                        
                        if err_snap < 1e-6:
                            best_rank = r
                            best_solution = (U_snap, V_snap, W_snap)
                            found_valid = True
                            break
                        
            except Exception:
                pass
        
        if found_valid:
            break
    
    U_res, V_res, W_res = best_solution
    
    return {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
