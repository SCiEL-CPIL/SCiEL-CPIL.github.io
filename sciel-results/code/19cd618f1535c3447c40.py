import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target Tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def verify(U, V, W, T, tol=1e-6):
        R = U.shape[1]
        try:
            # U: (nm, R), V: (mp, R), W: (pn, R)
            # T_approx[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
            # Efficient computation:
            # Temp[u, v, r] = U[u, r] * V[v, r]
            # T_approx[u, v, w] = sum_r Temp[u, v, r] * W[w, r]
            
            # Using einsum
            # 'ur,vr->uvr'
            UV = np.einsum('ur,vr->uvr', U, V)
            T_approx = np.einsum('uvr,wr->uvw', UV, W)
            
            if np.allclose(T_approx, T, atol=tol, rtol=0):
                return True
            return False
        except Exception:
            return False

    def calculate_error(U, V, W):
        try:
            UV = np.einsum('ur,vr->uvr', U, V)
            T_approx = np.einsum('uvr,wr->uvw', UV, W)
            err = np.sum((T_approx - T)**2)
            return err
        except Exception:
            return 1e10

    def discrete_local_search(U_init, V_init, W_init, T, max_iters=1000, time_limit=5.0):
        """
        Performs local search on the discrete grid of half-integers.
        U, V, W are arrays of half-integers (multiplied by 2 for integer arithmetic).
        """
        U = np.array(U_init, dtype=float) * 2
        V = np.array(V_init, dtype=float) * 2
        W = np.array(W_init, dtype=float) * 2
        
        current_err = calculate_error(U/2, V/2, W/2)
        start_time = time.time()
        
        # Precompute indices for fast updates?
        # For simplicity, we just recompute. Dimensions are small enough (6, 18, 12, R=18).
        # UV is 6x18x18 = 1944. T_approx is 6x18x12 = 1296.
        # Recomputation is fast.
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random matrix and a random entry
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                rows, cols = U.shape
            elif mat_idx == 1:
                mat = V
                rows, cols = V.shape
            else:
                mat = W
                rows, cols = W.shape
            
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, cols - 1)
            
            old_val = mat[r_idx, c_idx]
            # Try changing by +/- 1 (which is +/- 0.5 in original scale)
            delta = random.choice([-1, 1])
            new_val = old_val + delta
            
            mat[r_idx, c_idx] = new_val
            
            if mat_idx == 0:
                new_err = calculate_error(U/2, V/2, W/2)
            elif mat_idx == 1:
                new_err = calculate_error(U/2, V/2, W/2)
            else:
                new_err = calculate_error(U/2, V/2, W/2)
            
            if new_err < current_err:
                current_err = new_err
                # Accept
                # Optional: check if solved
                if current_err < 1e-9:
                    return U/2, V/2, W/2, True
            else:
                # Revert
                mat[r_idx, c_idx] = old_val
        
        # Check if solved at the end
        if current_err < 1e-9:
            return U/2, V/2, W/2, True
        return U/2, V/2, W/2, False

    def try_find_rank(target_rank, time_limit):
        T_local = T
        nm, mp, pn = n*m, m*p, p*n
        
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        start_time = time.time()
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random real numbers in [-1, 1]
            # Scaling helps optimization
            U = np.random.uniform(-1, 1, (nm, target_rank))
            V = np.random.uniform(-1, 1, (mp, target_rank))
            W = np.random.uniform(-1, 1, (pn, target_rank))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_x = x[:nm*target_rank].reshape(nm, target_rank)
                V_x = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_x = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                return calculate_error(U_x, V_x, W_x)
            
            # Use Nelder-Mead or Powell
            # Powell is often better for non-smooth or noisy, but Nelder-Mead is derivative-free and robust
            # Let's use Powell with more iterations
            res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
            
            U_opt = res.x[:nm*target_rank].reshape(nm, target_rank)
            V_opt = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W_opt = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            
            err = res.fun
            if err < best_err:
                best_err = err
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
            
            # Round to nearest half-integer
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if verify(U_round, V_round, W_round, T_local, tol=1e-6):
                return U_round, V_round, W_round
            
            # If close, try discrete local search
            if err < 1.0:
                U_ls, V_ls, W_ls, found = discrete_local_search(U_round, V_round, W_round, T_local, max_iters=500, time_limit=2.0)
                if found:
                    return U_ls, V_ls, W_ls
        
        # If we didn't find an exact solution, return the best continuous solution rounded
        # But we need an exact solution to return.
        # If no exact solution found, we must fall back or try higher rank.
        return None

    time_limit_total = 140
    t_start = time.time()
    
    # Try rank 18
    # Lower bound is 18.
    res_18 = try_find_rank(18, 60)
    if res_18:
        U_d, V_d, W_d = res_18
        return {
            "U": U_d.tolist(),
            "V": V_d.tolist(),
            "W": W_d.tolist()
        }
        
    # Try rank 19
    if time.time() - t_start < 100:
        res_19 = try_find_rank(19, 30)
        if res_19:
            U_d, V_d, W_d = res_19
            return {
                "U": U_d.tolist(),
                "V": V_d.tolist(),
                "W": W_d.tolist()
            }

    # Try rank 20
    if time.time() - t_start < 120:
        res_20 = try_find_rank(20, 15)
        if res_20:
            U_d, V_d, W_d = res_20
            return {
                "U": U_d.tolist(),
                "V": V_d.tolist(),
                "W": W_d.tolist()
            }

    # Fallback: Schoolbook
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
