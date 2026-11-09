import numpy as np
import time
import random
from scipy.optimize import minimize, minimize_scalar
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    def compute_error(U, V, W, r):
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        error = np.sum((T - target) ** 2)
        return error

    err0 = compute_error(U0, V0, W0, rank_schoolbook)
    if err0 < 1e-9:
        best_error = err0
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook

    def polish(U, V, W, r, max_iter=500):
        """Local search on half-integer grid"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        current_err = compute_error(U, V, W, r)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if current_err < 1e-12:
                break
                
            improved = False
            # Try small perturbations
            for _ in range(100):
                idx = random.randint(0, 2)
                if idx == 0:
                    row, col = random.randint(0, n*m-1), random.randint(0, r-1)
                    delta = random.choice([-0.5, 0.5])
                    U_new = U.copy()
                    U_new[row, col] += delta
                    err = compute_error(U_new, V, W, r)
                    if err < current_err:
                        U = U_new
                        current_err = err
                        improved = True
                elif idx == 1:
                    row, col = random.randint(0, m*p-1), random.randint(0, r-1)
                    delta = random.choice([-0.5, 0.5])
                    V_new = V.copy()
                    V_new[row, col] += delta
                    err = compute_error(U, V_new, W, r)
                    if err < current_err:
                        V = V_new
                        current_err = err
                        improved = True
                else:
                    row, col = random.randint(0, p*n-1), random.randint(0, r-1)
                    delta = random.choice([-0.5, 0.5])
                    W_new = W.copy()
                    W_new[row, col] += delta
                    err = compute_error(U, V, W_new, r)
                    if err < current_err:
                        W = W_new
                        current_err = err
                        improved = True
            
            if not improved:
                break
        
        return U, V, W, current_err

    # Try rank 17 with multiple strategies
    target_rank = 17
    if target_rank < best_rank:
        for restart in range(30):
            if time.time() > deadline - 15:
                break
                
            # Different initialization strategies
            if restart % 3 == 0:
                U = np.random.uniform(-2, 2, size=(n*m, target_rank))
                V = np.random.uniform(-2, 2, size=(m*p, target_rank))
                W = np.random.uniform(-2, 2, size=(p*n, target_rank))
            elif restart % 3 == 1:
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
            else:
                U = np.random.uniform(-3, 3, size=(n*m, target_rank))
                V = np.random.uniform(-3, 3, size=(m*p, target_rank))
                W = np.random.uniform(-3, 3, size=(p*n, target_rank))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:n*m*target_rank + m*p*target_rank], x[n*m*target_rank + m*p*target_rank:]
                U_curr = U_flat.reshape(n*m, target_rank)
                V_curr = V_flat.reshape(m*p, target_rank)
                W_curr = W_flat.reshape(p*n, target_rank)
                return compute_error(U_curr, V_curr, W_curr, target_rank)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            except Exception:
                continue
                
            U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Polish on half-integer grid
            U_pol, V_pol, W_pol, err_pol = polish(U_opt, V_opt, W_opt, target_rank)
            
            if err_pol < best_error:
                best_error = err_pol
                best_U, best_V, best_W = U_pol, V_pol, W_pol
                best_rank = target_rank
                
                if best_error < 1e-9:
                    break

    # If rank 17 not found, try rank 16
    if best_rank > 16 and time.time() < deadline - 25:
        target_rank = 16
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            U = np.random.uniform(-2, 2, size=(n*m, target_rank))
            V = np.random.uniform(-2, 2, size=(m*p, target_rank))
            W = np.random.uniform(-2, 2, size=(p*n, target_rank))
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:n*m*target_rank + m*p*target_rank], x[n*m*target_rank + m*p*target_rank:]
                U_curr = U_flat.reshape(n*m, target_rank)
                V_curr = V_flat.reshape(m*p, target_rank)
                W_curr = W_flat.reshape(p*n, target_rank)
                return compute_error(U_curr, V_curr, W_curr, target_rank)
                
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-5})
            except Exception:
                continue
                
            U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            U_pol, V_pol, W_pol, err_pol = polish(U_opt, V_opt, W_opt, target_rank)
            
            if err_pol < best_error:
                best_error = err_pol
                best_U, best_V, best_W = U_pol, V_pol, W_pol
                best_rank = target_rank
                if best_error < 1e-9:
                    break

    # Final cleanup: ensure half-integers
    def to_half_int(arr):
        return np.round(arr * 2) / 2

    U_final = to_half_int(best_U)
    V_final = to_half_int(best_V)
    W_final = to_half_int(best_W)
    
    # Verify and clean
    err_final = compute_error(U_final, V_final, W_final, best_rank)
    if err_final > 1e-6:
        # Fall back to schoolbook if something went wrong
        U_final = U0
        V_final = V0
        W_final = W0
        best_rank = rank_schoolbook

    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
