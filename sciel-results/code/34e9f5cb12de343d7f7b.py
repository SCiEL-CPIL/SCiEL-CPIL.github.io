import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if multiplying A[i,j] and B[j,k] contributes to C[i,k]
    # In the decomposition sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
    # Wait, the standard convention is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) such that T[i,j,k] = 1.
    # The factorization is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This matches the prompt's W[k*n+i, r] convention.
    
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
        # Reconstruct tensor
        # T_recon[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T_recon = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            # Reshape to match indices
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n) # Note: W index is k*n+i, so w_mat[k, i]
            
            # The term is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is an element-wise product summed over c.
            # We can compute this efficiently using einsum or loops.
            # For small sizes, loops are fine.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        
        return np.allclose(T_recon, T, atol=tol)

    def snap_to_half_int(arr):
        return np.round(arr * 2) / 2

    # Try to find a lower rank
    # Time limit guard
    start_time = time.time()
    deadline = 110.0 # 10 seconds margin
    
    # Try ranks from a reasonable lower bound up to schoolbook - 1
    # Lower bound is max(n*m, m*p, p*n) = 21
    # Let's try a few specific ranks
    target_ranks = [30, 35, 40, 45, 50, 55, 60]
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        found = False
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W randomly
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            def objective(params):
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Compute error
                err = 0.0
                for c in range(r):
                    u_mat = U_flat[:, c].reshape(n, m)
                    v_mat = V_flat[:, c].reshape(m, p)
                    w_mat = W_flat[:, c].reshape(p, n)
                    # Add to error tensor
                    # To speed up, we can just compute the norm difference
                    # But we need the full tensor to compare with T
                    # Let's accumulate into a temporary tensor
                    # For efficiency in the loop, we might just compute the scalar error
                    # err = sum (T - T_recon)^2
                    # This is expensive if done from scratch each time.
                    # Let's use a vectorized approach if possible.
                    pass
                
                # Efficient calculation:
                # T_recon[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                # This is a 3-way contraction.
                # We can use np.einsum
                # T_recon = np.einsum('ic,jc,kc->ijk', U_flat, V_flat, W_flat) 
                # Wait, indices:
                # U is (n*m, r) -> index i*m+j
                # V is (m*p, r) -> index j*p+k
                # W is (p*n, r) -> index k*n+i
                # This einsum doesn't directly map to i,j,k without reshaping.
                
                u_reshaped = U_flat.reshape(n, m, r)
                v_reshaped = V_flat.reshape(m, p, r)
                w_reshaped = W_flat.reshape(p, n, r)
                
                # T_recon[i,j,k] = sum_c u_reshaped[i,j,c] * v_reshaped[j,k,c] * w_reshaped[k,i,c]
                T_recon = np.einsum('ijc,jkc,kic->ijk', u_reshaped, v_reshaped, w_reshaped)
                
                diff = T_recon - T
                return np.sum(diff**2)

            params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, params_init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Check exactness after snapping
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                U_snapped = snap_to_half_int(U_opt)
                V_snapped = snap_to_half_int(V_opt)
                W_snapped = snap_to_half_int(W_opt)
                
                if verify(U_snapped, V_snapped, W_snapped):
                    best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                    best_rank = r
                    found = True
                    break # Found for this rank, try lower? No, we iterate ranks.
        
        # If found a valid rank, we can stop searching higher ranks (since we go low to high? No, list is high to low or low to high?)
        # My list is [30, 35, ... 60]. I want the MINIMAL rank.
        # So I should iterate from low to high.
        # If I find one at 30, I should return it.
        # If I don't find one at 30, I try 35, etc.
        if found:
            break

    # If no better rank found, return schoolbook
    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
