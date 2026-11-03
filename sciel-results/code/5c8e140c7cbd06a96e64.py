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
# We'll generate these more efficiently
all_I = np.repeat(np.arange(num_I), num_J * num_K)
all_J = np.tile(np.repeat(np.arange(num_J), num_K), num_I)
all_K = np.tile(np.arange(num_K), num_I * num_J)

# Create mask for on-positions
on_set = set(zip(on_I.tolist(), on_J.tolist(), on_K.tolist()))
off_mask = np.array([
    (all_I[s], all_J[s], all_K[s]) not in on_set
    for s in range(len(all_I))
])
off_I = all_I[off_mask]
off_J = all_J[off_mask]
off_K = all_K[off_mask]
num_off = len(off_I)  # 3906

def compute_error(U, V, W, r):
    """
    Compute the sum of squared errors for the full tensor decomposition.
    U: (num_I, r), V: (num_J, r), W: (num_K, r)
    """
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

def compute_error_grad(U, V, W, r):
    """
    Compute the gradient of the error with respect to U, V, W.
    """
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    # For on-positions
    for t in range(r):
        T_on_t = U[on_I, t] * V[on_J, t] * W[on_K, t]
        dU[on_I, t] += 2 * (T_on_t - 1) * V[on_J, t] * W[on_K, t]
        dV[on_J, t] += 2 * (T_on_t - 1) * U[on_I, t] * W[on_K, t]
        dW[on_K, t] += 2 * (T_on_t - 1) * U[on_I, t] * V[on_J, t]
    
    # For off-positions
    for t in range(r):
        T_off_t = U[off_I, t] * V[off_J, t] * W[off_K, t]
        dU[off_I, t] += 2 * T_off_t * V[off_J, t] * W[off_K, t]
        dV[off_J, t] += 2 * T_off_t * U[off_I, t] * W[off_K, t]
        dW[off_K, t] += 2 * T_off_t * U[off_I, t] * V[off_J, t]
    
    return dU, dV, dW

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin below 150s kill
    
    # Schoolbook solution as fallback (rank 63)
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
    
    # Strategy: Iterative rank reduction
    # Start with rank 63 (schoolbook), try to reduce to 62, 61, ...
    # At each step, try to set one column to zero and re-optimize
    
    from scipy.optimize import minimize
    
    # Try ranks from 62 down to 50
    for target_rank in range(62, 49, -1):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Use the best solution from the previous rank (or schoolbook for 62)
        if target_rank == 62:
            U_curr, V_curr, W_curr = U_school, V_school, W_school
            curr_rank = 63
        else:
            U_curr, V_curr, W_curr = best_solution
            curr_rank = best_rank
        
        if curr_rank != target_rank + 1:
            # Can't reduce from current rank to target_rank
            # Try from schoolbook
            U_curr, V_curr, W_curr = U_school, V_school, W_school
            curr_rank = 63
        
        # Try multiple strategies to reduce rank by 1
        found = False
        
        # Strategy 1: Try each column as the one to zero out
        for zero_col in range(curr_rank):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Create initial solution with one column zeroed
            U_init = U_curr.copy()
            V_init = V_curr.copy()
            W_init = W_curr.copy()
            
            # Remove the zero_col-th column
            mask = np.ones(curr_rank, dtype=bool)
            mask[zero_col] = False
            U_init = U_curr[:, mask]
            V_init = V_curr[:, mask]
            W_init = W_curr[:, mask]
            
            # Add small noise to help escape local minima
            U_init = U_init + 0.01 * np.random.randn(num_I, target_rank)
            V_init = V_init + 0.01 * np.random.randn(num_J, target_rank)
            W_init = W_init + 0.01 * np.random.randn(num_K, target_rank)
            
            # Optimize
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U = params[0 : num_I*target_rank].reshape(num_I, target_rank)
                V = params[num_I*target_rank : num_I*target_rank + num_J*target_rank].reshape(num_J, target_rank)
                W = params[num_I*target_rank + num_J*target_rank : num_I*target_rank + num_J*target_rank + num_K*target_rank].reshape(num_K, target_rank)
                return compute_error(U, V, W, target_rank)
            
            def objective_grad(params):
                U = params[0 : num_I*target_rank].reshape(num_I, target_rank)
                V = params[num_I*target_rank : num_I*target_rank + num_J*target_rank].reshape(num_J, target_rank)
                W = params[num_I*target_rank + num_J*target_rank : num_I*target_rank + num_J*target_rank + num_K*target_rank].reshape(num_K, target_rank)
                dU, dV, dW = compute_error_grad(U, V, W, target_rank)
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            try:
                res = minimize(objective, params0, jac=objective_grad, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6})
                
                if res.fun < 1e-8:
                    params_opt = res.x
                    U_opt = params_opt[0 : num_I*target_rank].reshape(num_I, target_rank)
                    V_opt = params_opt[num_I*target_rank : num_I*target_rank + num_J*target_rank].reshape(num_J, target_rank)
                    W_opt = params_opt[num_I*target_rank + num_J*target_rank : num_I*target_rank + num_J*target_rank + num_K*target_rank].reshape(num_K, target_rank)
                    
                    # Verify
                    err = compute_error(U_opt, V_opt, W_opt, target_rank)
                    if err < 1e-6:
                        # Try to snap to integers/half-integers
                        U_snap = np.round(U_opt * 2) / 2
                        V_snap = np.round(V_opt * 2) / 2
                        W_snap = np.round(W_opt * 2) / 2
                        
                        # Verify snapped solution
                        err_snap = compute_error(U_snap, V_snap, W_snap, target_rank)
                        
                        if err_snap < 1e-6:
                            best_rank = target_rank
                            best_solution = (U_snap, V_snap, W_snap)
                            found = True
                            break
                        else:
                            # Use the non-snapped solution
                            best_rank = target_rank
                            best_solution = (U_opt, V_opt, W_opt)
                            found = True
                            break
                            
            except Exception:
                pass
        
        if found:
            continue  # Try to reduce further
        else:
            break  # Couldn't reduce to this rank, stop
    
    U_res, V_res, W_res = best_solution
    
    return {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
