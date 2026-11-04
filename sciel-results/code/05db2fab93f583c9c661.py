import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if multiplying A[i,j] and B[j,k] contributes to C[i,k]
    # In the decomposition sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for c in range(r):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            T_recon += u_mat[:, :, None] * v_mat[None, :, :] * w_mat[None, None, :] # Broadcasting: u[i,j,1] * v[1,j,k] * w[1,1,i] -> [i,j,k]?
            # Wait, w_mat is (p, n) indexed by [k, i].
            # u_mat[i, j], v_mat[j, k], w_mat[k, i].
            # T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # Vectorized:
            # u_mat: (n, m)
            # v_mat: (m, p)
            # w_mat: (p, n)
            # We want sum over c of u_c[i,j] v_c[j,k] w_c[k,i]
            # This is not a simple outer product.
            # Let's stick to the loop for clarity and correctness in verify, it's only called at the end.
            # But in the objective function, we need speed.
            pass
        
        # Re-implement verify with loops for safety, or efficient einsum
        T_recon = np.einsum('ic,jc,kc->ijk', 
                            U.reshape(n, m, r), 
                            V.reshape(m, p, r), 
                            W.reshape(p, n, r))
        return np.allclose(T_recon, T, atol=tol)

    def snap_to_rational(arr, max_denom=10):
        # Try to snap to rationals with small denominators
        # For now, stick to half-integers as per original, but maybe add integers
        # The original snap was: np.round(arr * 2) / 2
        # Let's try integers first, then half-integers
        int_snap = np.round(arr)
        if np.allclose(arr, int_snap, atol=1e-6):
            return int_snap
        half_snap = np.round(arr * 2) / 2
        if np.allclose(arr, half_snap, atol=1e-6):
            return half_snap
        return arr

    start_time = time.time()
    deadline = 140.0 # 10 seconds margin before 150s kill

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Search ranks from 62 down to 50 (or lower)
    # If we find a valid rank, we want the MINIMAL rank.
    # So we should search 50, 51, ..., 62.
    # But finding 50 is harder than 62.
    # If we find 62, we have a score > 1.0.
    # If we find 50, we have a much better score.
    # Given time constraints, finding 62 is more likely.
    # Let's iterate 62, 61, 60... and keep the best (lowest) found.
    # But if we find 60, we should stop? No, we want minimal.
    # If we find 60, we should try 59?
    # Let's just try a few ranks and keep the best valid one.
    
    target_ranks = [62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        found = False
        # Try a few random initializations / DE runs
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Define bounds
            bounds = [(-2.0, 2.0)] * (n*m*r + m*p*r + p*n*r)
            
            def objective(params):
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                u_reshaped = U_flat.reshape(n, m, r)
                v_reshaped = V_flat.reshape(m, p, r)
                w_reshaped = W_flat.reshape(p, n, r)
                
                T_recon = np.einsum('ijc,jkc,kic->ijk', u_reshaped, v_reshaped, w_reshaped)
                
                diff = T_recon - T
                return np.sum(diff**2)

            try:
                # Use Differential Evolution for global search
                # maxiter and tol control convergence
                res = differential_evolution(
                    objective, 
                    bounds, 
                    maxiter=100, 
                    tol=1e-6, 
                    popsize=15,
                    seed=42+attempt
                )
                
                if res.fun < 1e-4:
                    # Polish with Nelder-Mead
                    params_init = res.x
                    res2 = minimize(objective, params_init, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                    
                    if res2.fun < 1e-8:
                        U_opt = res2.x[:n*m*r].reshape(n*m, r)
                        V_opt = res2.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                        W_opt = res2.x[n*m*r + m*p*r:].reshape(p*n, r)
                        
                        # Try snapping
                        U_snapped = snap_to_rational(U_opt)
                        V_snapped = snap_to_rational(V_opt)
                        W_snapped = snap_to_rational(W_opt)
                        
                        if verify(U_snapped, V_snapped, W_snapped):
                            best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                            best_rank = r
                            found = True
                            break # Found for this rank, try lower ranks?
                            # Since we iterate high to low in target_ranks list?
                            # My list is [62, 61, ... 50].
                            # If I find 62, I break.
                            # I should then try 61?
                            # The loop continues to next r (61).
                            # So I will find the minimal rank among those tested.
                            # But I should break the inner loop and continue outer.
                            # The `break` exits the `for attempt` loop.
                            # The outer loop continues to next r.
                            
            except Exception:
                continue

    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
