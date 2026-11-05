import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def _verify(U, V, W, R):
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            max_err = np.max(np.abs(T - 1.0))
            return max_err < 1e-8
        except:
            return False

    def _snap(x):
        """Snap to nearest half-integer"""
        return round(x * 2) / 2.0

    def _compute_error(U_t, V_t, W_t):
        """Compute max error from target tensor"""
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.max(np.abs(T - 1.0))

    def _local_search(U, V, W, max_iters=500):
        """
        Local search with three strategies:
        1. Coordinate descent on worst element
        2. Fix two factors, solve for third via least squares
        3. Random restarts within local neighborhood
        """
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        err = T - 1.0
        max_err = np.max(np.abs(err))
        
        if max_err < 1e-8:
            return U, V, W, max_err
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Strategy 1: Coordinate descent on worst element
            worst_idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
            wi, wj, wk = worst_idx
            werr = err[wi, wj, wk]
            
            # Try adjusting each factor at the worst position
            best_improvement = 0
            best_factor = None
            best_r = None
            best_new_val = None
            
            for r in range(R):
                # Try adjusting U
                grad_u = V_t[wj, wk, r] * W_t[wk, wi, r]
                if abs(grad_u) > 1e-10:
                    old_u = U_t[wi, wj, r]
                    # Optimal continuous value
                    optimal_u = old_u - werr / grad_u
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        c = _snap(old_u + delta)
                        if c != old_u:
                            U_t[wi, wj, r] = c
                            new_err = _compute_error(U_t, V_t, W_t)
                            improvement = max_err - new_err
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_factor = 'U'
                                best_r = r
                                best_new_val = c
                            U_t[wi, wj, r] = old_u
                
                # Try adjusting V
                grad_v = U_t[wi, wj, r] * W_t[wk, wi, r]
                if abs(grad_v) > 1e-10:
                    old_v = V_t[wj, wk, r]
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        c = _snap(old_v + delta)
                        if c != old_v:
                            V_t[wj, wk, r] = c
                            new_err = _compute_error(U_t, V_t, W_t)
                            improvement = max_err - new_err
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_factor = 'V'
                                best_r = r
                                best_new_val = c
                            V_t[wj, wk, r] = old_v
                
                # Try adjusting W
                grad_w = U_t[wi, wj, r] * V_t[wj, wk, r]
                if abs(grad_w) > 1e-10:
                    old_w = W_t[wk, wi, r]
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        c = _snap(old_w + delta)
                        if c != old_w:
                            W_t[wk, wi, r] = c
                            new_err = _compute_error(U_t, V_t, W_t)
                            improvement = max_err - new_err
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_factor = 'W'
                                best_r = r
                                best_new_val = c
                            W_t[wk, wi, r] = old_w
            
            if best_factor is not None:
                if best_factor == 'U':
                    U_t[wi, wj, best_r] = best_new_val
                elif best_factor == 'V':
                    V_t[wj, wk, best_r] = best_new_val
                else:
                    W_t[wk, wi, best_r] = best_new_val
                
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                err = T - 1.0
                max_err = np.max(np.abs(err))
                if max_err < 1e-8:
                    return U, V, W, max_err
                continue
            
            # Strategy 2: Fix two factors, solve for third via least squares
            if it % 3 == 0:
                # Solve for U given V, W
                # For each (i,j), we need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
                # This is: U[i,j,:] @ (V[j,:,r] * W[:,i,r]) = ones(p)
                # Let A[i,j,k,r] = V[j,k,r] * W[k,i,r], then U[i,j,:] @ A[i,j,:,:].T = ones(p)
                solved_any = False
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V_t[j, k, :] * W_t[k, i, :]
                        
                        # Solve least squares
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            u_snapped = np.array([_snap(x) for x in u_sol])
                            
                            # Check if this improves error
                            old_u = U_t[i, j, :].copy()
                            U_t[i, j, :] = u_snapped
                            new_err = _compute_error(U_t, V_t, W_t)
                            if new_err < max_err - 1e-10:
                                max_err = new_err
                                solved_any = True
                                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                                err = T - 1.0
                            else:
                                U_t[i, j, :] = old_u
                        except:
                            pass
                
                if solved_any:
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue
                
                # Solve for V given U, W
                solved_any = False
                for j in range(m):
                    for k in range(p):
                        # For each (i), sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                        A = np.zeros((n, R))
                        for i in range(n):
                            A[i, :] = U_t[i, j, :] * W_t[k, i, :]
                        
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            v_snapped = np.array([_snap(x) for x in v_sol])
                            
                            old_v = V_t[j, k, :].copy()
                            V_t[j, k, :] = v_snapped
                            new_err = _compute_error(U_t, V_t, W_t)
                            if new_err < max_err - 1e-10:
                                max_err = new_err
                                solved_any = True
                                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                                err = T - 1.0
                            else:
                                V_t[j, k, :] = old_v
                        except:
                            pass
                
                if solved_any:
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue
                
                # Solve for W given U, V
                solved_any = False
                for k in range(p):
                    for i in range(n):
                        # For each (j), sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                        A = np.zeros((m, R))
                        for j in range(m):
                            A[j, :] = U_t[i, j, :] * V_t[j, k, :]
                        
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            w_snapped = np.array([_snap(x) for x in w_sol])
                            
                            old_w = W_t[k, i, :].copy()
                            W_t[k, i, :] = w_snapped
                            new_err = _compute_error(U_t, V_t, W_t)
                            if new_err < max_err - 1e-10:
                                max_err = new_err
                                solved_any = True
                                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                                err = T - 1.0
                            else:
                                W_t[k, i, :] = old_w
                        except:
                            pass
                
                if solved_any:
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue
            
            # Strategy 3: Random perturbation
            if np.random.random() < 0.15:
                for _ in range(5):
                    factor = np.random.randint(3)
                    r_rand = np.random.randint(R)
                    
                    if factor == 0:
                        idx = np.random.randint(n * m)
                        i, j = divmod(idx, m)
                        old_val = U_t[i, j, r_rand]
                        new_val = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                        U_t[i, j, r_rand] = new_val
                        new_err = _compute_error(U_t, V_t, W_t)
                        if new_err < max_err - 1e-10:
                            max_err = new_err
                            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                            err = T - 1.0
                        else:
                            U_t[i, j, r_rand] = old_val
                    
                    elif factor == 1:
                        idx = np.random.randint(m * p)
                        j, k = divmod(idx, p)
                        old_val = V_t[j, k, r_rand]
                        new_val = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                        V_t[j, k, r_rand] = new_val
                        new_err = _compute_error(U_t, V_t, W_t)
                        if new_err < max_err - 1e-10:
                            max_err = new_err
                            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                            err = T - 1.0
                        else:
                            V_t[j, k, r_rand] = old_val
                    
                    else:
                        idx = np.random.randint(p * n)
                        k, i = divmod(idx, n)
                        old_val = W_t[k, i, r_rand]
                        new_val = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                        W_t[k, i, r_rand] = new_val
                        new_err = _compute_error(U_t, V_t, W_t)
                        if new_err < max_err - 1e-10:
                            max_err = new_err
                            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                            err = T - 1.0
                        else:
                            W_t[k, i, r_rand] = old_val
                
                if max_err < 1e-8:
                    return U, V, W, max_err

        return U, V, W, max_err

    # Try ranks from 24 to 35
    # We prioritize lower ranks but spend limited time on each
    # Rank 24 is the target.
    for R in range(24, rank_school):
        if time.time() > deadline:
            break
        
        # Time allocation: 6 seconds per rank
        time_limit = time.time() + 6.0
        
        num_trials = 30
        for trial in range(num_trials):
            if time.time() > time_limit:
                break
            
            # Initialize with random half-integers
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, R))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, R))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, R))
            
            U, V, W, err = _local_search(U, V, W, max_iters=300)
            
            if err < 1e-8:
                valid = _verify(U, V, W, R)
                if valid:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    U_list = [[_snap(x) for x in row] for row in U_list]
                    V_list = [[_snap(x) for x in row] for row in V_list]
                    W_list = [[_snap(x) for x in row] for row in W_list]
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
