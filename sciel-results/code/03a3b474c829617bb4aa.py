import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    # Target tensor T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p))
    
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        # result[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)

    def verify_and_round(U, V, W, target_rank):
        """Round to half-integers and verify exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness
        err = compute_error(U_r, V_r, W_r, target_rank)
        if err < 1e-9:
            return U_r.reshape(-1, target_rank), V_r.reshape(-1, target_rank), W_r.reshape(-1, target_rank)
        return None

    def polish_solution(U, V, W, target_rank, max_iters=200):
        """Polish the solution using Least Squares."""
        
        def objective(params):
            U_flat = params[:n*m*target_rank].reshape(n, m, target_rank)
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p, n, target_rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_flat, V_flat, W_flat)
            diff = result - T
            return diff.flatten()
        
        initial_params = np.concatenate([
            U.flatten(), 
            V.flatten(), 
            W.flatten()
        ])
        
        try:
            # Use 'lm' for speed, but also try 'trf' if 'lm' fails or converges slowly
            res = opt.least_squares(
                objective, 
                initial_params, 
                method='lm', 
                max_nfev=max_iters,
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15
            )
            
            if res.cost < 1e-15:
                U_new = res.x[:n*m*target_rank].reshape(n, m, target_rank)
                V_new = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
                W_new = res.x[n*m*target_rank + m*p*target_rank:].reshape(p, n, target_rank)
                return U_new.reshape(-1, target_rank), V_new.reshape(-1, target_rank), W_new.reshape(-1, target_rank)
        except:
            pass
            
        # Fallback to TRF if LM fails
        try:
            res = opt.least_squares(
                objective, 
                initial_params, 
                method='trf', 
                max_nfev=max_iters,
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15
            )
            if res.cost < 1e-15:
                U_new = res.x[:n*m*target_rank].reshape(n, m, target_rank)
                V_new = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
                W_new = res.x[n*m*target_rank + m*p*target_rank:].reshape(p, n, target_rank)
                return U_new.reshape(-1, target_rank), V_new.reshape(-1, target_rank), W_new.reshape(-1, target_rank)
        except:
            pass
            
        return U, V, W

    def als_decompose(target_rank, max_iters=50, tol=1e-10, seed=None, polish=True):
        """Alternating Least Squares for tensor decomposition."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        # Initialize with small random values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        prev_err = float('inf')
        
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return None, None, None, None
                
            # --- Vectorized Update for U ---
            # Reshape V and W to 3D
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # A[i, j, k, r] = V[j, k, r] * W[k, i, r]
            # Shape: (n, m, p, R)
            A_U = np.einsum('jkr,kir->ijk r', V_r, W_r)
            
            # T is (n, m, p). We need to solve A_U[i, j, :, :] @ U[i, j, :] = T[i, j, :]
            # Reshape A_U to (n*m, p, R) and T to (n*m, p)
            A_U_flat = A_U.reshape(n * m, p, target_rank)
            T_flat = T.reshape(n * m, p)
            
            # Solve for each i*m+j
            U_new_flat = np.zeros_like(U)
            for idx in range(n * m):
                A = A_U_flat[idx]
                t = T_flat[idx]
                try:
                    U_new_flat[idx] = np.linalg.lstsq(A, t, rcond=None)[0]
                except:
                    U_new_flat[idx] = U[idx]
            U = U_new_flat

            # --- Vectorized Update for V ---
            U_r = U.reshape(n, m, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # B[j, k, i, r] = U[i, j, r] * W[k, i, r]
            # Shape: (m, p, n, R)
            A_V = np.einsum('ijr,kir->jkir', U_r, W_r)
            
            # T is (n, m, p). We need B[j, k, :, :] @ V[j, k, :] = T[:, j, k]
            # Reshape A_V to (m*p, n, R) and T to (n, m, p) -> (m*p, n) by transposing T
            # T_transposed[j, k, i] = T[i, j, k]
            T_trans = np.transpose(T, (1, 2, 0)) # (m, p, n)
            
            A_V_flat = A_V.reshape(m * p, n, target_rank)
            T_trans_flat = T_trans.reshape(m * p, n)
            
            V_new_flat = np.zeros_like(V)
            for idx in range(m * p):
                A = A_V_flat[idx]
                t = T_trans_flat[idx]
                try:
                    V_new_flat[idx] = np.linalg.lstsq(A, t, rcond=None)[0]
                except:
                    V_new_flat[idx] = V[idx]
            V = V_new_flat

            # --- Vectorized Update for W ---
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            
            # C[k, i, j, r] = U[i, j, r] * V[j, k, r]
            # Shape: (p, n, m, R)
            A_W = np.einsum('ijr,jkr->kijr', U_r, V_r)
            
            # T is (n, m, p). We need C[k, i, :, :] @ W[k, i, :] = T[i, :, k]
            # Reshape A_W to (p*n, m, R) and T to (p, n, m) by transposing T
            # T_trans2[i, k, j] = T[i, j, k] -> We want index [k, i, j]
            # T_trans2[k, i, j] = T[i, j, k]
            T_trans2 = np.transpose(T, (2, 0, 1)) # (p, n, m)
            
            A_W_flat = A_W.reshape(p * n, m, target_rank)
            T_trans2_flat = T_trans2.reshape(p * n, m)
            
            W_new_flat = np.zeros_like(W)
            for idx in range(p * n):
                A = A_W_flat[idx]
                t = T_trans2_flat[idx]
                try:
                    W_new_flat[idx] = np.linalg.lstsq(A, t, rcond=None)[0]
                except:
                    W_new_flat[idx] = W[idx]
            W = W_new_flat
            
            # Check convergence
            err = compute_error(U, V, W, target_rank)
            if err < tol:
                if polish:
                    U, V, W = polish_solution(U, V, W, target_rank)
                    err = compute_error(U, V, W, target_rank)
                return U, V, W, err
            if abs(err - prev_err) < 1e-12:
                if err < 1e-6:
                    if polish:
                        U, V, W = polish_solution(U, V, W, target_rank)
                        err = compute_error(U, V, W, target_rank)
                    return U, V, W, err
                break
            prev_err = err
            
        # Final polish if we are close
        err = compute_error(U, V, W, target_rank)
        if err < 1e-3 and polish:
            U, V, W = polish_solution(U, V, W, target_rank)
            err = compute_error(U, V, W, target_rank)
            
        return U, V, W, err
    
    # Try ranks from 14 upward (known minimal rank for 2x3x3 is 14)
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
            
        found = False
        # Increase restarts for lower ranks, as they are harder but more valuable
        max_restarts = 300 if target_rank <= 14 else 100
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            U, V, W, err = als_decompose(target_rank, max_iters=30, tol=1e-8, seed=restart, polish=True)
            
            if U is None:
                break
                
            if err < 1e-8:
                # Attempt to round and verify
                rounded = verify_and_round(U, V, W, target_rank)
                if rounded is not None:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    found = True
                    break
                # If rounding fails, try to polish more or accept float? 
                # Task requires integer/half-integer. So we must find one that rounds well.
                # If err is small, it should round well.
                
        if found:
            continue
        else:
            continue
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
