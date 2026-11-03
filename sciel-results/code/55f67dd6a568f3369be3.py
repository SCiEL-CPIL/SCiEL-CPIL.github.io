"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Build the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find a lower rank decomposition via random search + refinement
    # We'll try ranks from 28 up to 55
    for target_rank in range(28, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating least squares to fit the tensor
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Fix U, V, solve for W
                # For each (a,b) in target, W[a,b,:] should satisfy:
                # target[a,b,:] = U[a,:] @ (V[b,:] * W[a,b,:])  -- this is not quite right
                
                # Actually, the tensor equation is:
                # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # For fixed U, V: for each c, W[c,:] = (sum over a,b) ... 
                # This is a linear system.
                
                # Let's use a different approach: vectorize
                # T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
                # In matrix form: T.reshape(n*m, m*p*p*n) = sum_t U[:,t] ⊗ (V[:,t] ⊗ W[:,t])
                
                # Simpler: use the fact that for fixed U and V, W can be solved
                # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # For each c: T[:, :, c] = sum_t (U[:,t] ⊗ V[:,t]) W[c,t]
                # So T_reshaped[c] = M @ W[c,:] where M[a,b,t] = U[a,t]*V[b,t]
                
                M = np.einsum('at,bt->abt', U, V)  # shape (n*m, m*p, target_rank)
                M_reshaped = M.reshape(n*m*m*p, target_rank)
                T_reshaped = target.reshape(n*m*m*p, p*n)  # shape (n*m*m*p, p*n)
                
                # T_reshaped = M_reshaped @ W  where W has shape (target_rank, p*n)
                # So W = solve(M_reshaped^T M_reshaped, M_reshaped^T T_reshaped)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_reshaped, T_reshaped, rcond=None)
                except:
                    continue
                W = W_new.T  # shape (p*n, target_rank)
                
                # Check convergence
                T_recon = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.linalg.norm(T_recon - target)
                if err < 1e-6:
                    # Found a valid decomposition
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
                
                if time.time() > deadline:
                    break
                
                # Fix U, W, solve for V
                M2 = np.einsum('at,ct->act', U, W)  # shape (n*m, p*n, target_rank)
                M2_reshaped = M2.reshape(n*m*p*n, target_rank)
                T_reshaped2 = target.reshape(n*m*p*n, m*p)  # need to reorder
                # T[a,c,b] = sum_t U[a,t] W[c,t] V[b,t]
                # So for each b: T[:,:,b] = sum_t (U[:,t] ⊗ W[:,t]) V[b,t]
                T_reorder = np.transpose(target, (0, 2, 1)).reshape(n*m*p*n, m*p)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M2_reshaped, T_reorder, rcond=None)
                except:
                    continue
                V = V_new.T  # shape (m*p, target_rank)
                
                # Fix V, W, solve for U
                M3 = np.einsum('bt,ct->bct', V, W)  # shape (m*p, p*n, target_rank)
                M3_reshaped = M3.reshape(m*p*p*n, target_rank)
                T_reorder2 = np.transpose(target, (1, 2, 0)).reshape(m*p*p*n, n*m)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M3_reshaped, T_reorder2, rcond=None)
                except:
                    continue
                U = U_new.T  # shape (n*m, target_rank)
            
            # Final check
            if time.time() > deadline:
                break
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.linalg.norm(T_recon - target)
            if err < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    # Convert to lists with rounding to nearest half-integer
    def to_half_int_list(arr):
        rounded = np.round(arr * 2) / 2
        # Verify
        T_recon = np.einsum('at,bt,ct->abc', 
                           np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2)
        if np.linalg.norm(T_recon - target) < 1e-4:
            return rounded.tolist()
        return arr.tolist()
    
    U_list = to_half_int_list(best_U)
    V_list = to_half_int_list(best_V)
    W_list = to_half_int_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
