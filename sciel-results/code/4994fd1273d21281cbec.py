"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    # Store non-zero indices for fast RHS calculation
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                nz_indices.append((a, b, c))
    
    # Convert to numpy array for fast indexing
    nz_a = np.array([x[0] for x in nz_indices])
    nz_b = np.array([x[1] for x in nz_indices])
    nz_c = np.array([x[2] for x in nz_indices])
    nz_val = np.ones(len(nz_indices))
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def compute_error(U, V, W):
        """Compute max absolute error between T and decomposition."""
        # T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # We can compute this efficiently by accumulating into a flat array or using einsum
        # einsum is usually fast enough for these small sizes
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - T_approx))
    
    def try_rank(target_rank, n_restarts=10, max_iter=200):
        """Try to find a rank-target_rank decomposition using fast ALS."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            # Scale to avoid numerical issues initially
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # --- Update U ---
                # We want to minimize || T - sum_r U[:,r] outer V[:,r] outer W[:,r] ||^2
                # This is equivalent to solving for each a:
                # T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
                # Let H_VW[r] = (V[:,r] outer W[:,r]).reshape(-1)  # shape (mp*pn,)
                # Then T_flat[a] = H_VW.T @ U[a,:]
                # So U = pinv(H_VW) @ T_flat.T  (where T_flat is (nm, mp*pn))
                
                # Efficiently construct H_VW: (target_rank, mp, pn) -> (target_rank, mp*pn)
                # H_VW[r, b, c] = V[b, r] * W[c, r]
                H_VW = (V[:, :, None] * W[None, :, :]).reshape(target_rank, mp * pn)
                
                # Compute RHS: b_U = H_VW.T @ T_flat.T
                # T_flat is (nm, mp*pn). T_flat.T is (mp*pn, nm).
                # b_U will be (target_rank, nm).
                # b_U[r, a] = sum_{b,c} H_VW[r, b, c] * T[a, b, c]
                # Since T is sparse, we can compute this faster.
                # H_VW_sparse = H_VW[:, nz_b * pn + nz_c]  # (target_rank, num_nz)
                # T_sparse = T[:, nz_b, nz_c] -> this is (nm, num_nz) but we know the values are 1
                # Actually, T[a, b, c] is 1 only if (a,b,c) is in nz_indices.
                # For a fixed r, b_U[r, a] = sum_{(b,c) in nz for this a? No, nz is global}
                # Let's just use the dense matrix multiplication if it's fast enough.
                # mp*pn = 784. target_rank ~ 90. nm = 16.
                # H_VW.T (90, 784) @ T_flat.T (784, 16) -> (90, 16).
                # This is very fast.
                
                T_flat = T.reshape(nm, mp * pn)
                RHS_U = H_VW.T @ T_flat.T  # (target_rank, nm)
                
                # Compute LHS: G_UU = H_VW.T @ H_VW  # (target_rank, target_rank)
                G_UU = H_VW.T @ H_VW
                
                # Solve G_UU @ U.T = RHS_U
                # U.T is (target_rank, nm)
                try:
                    U_new_T = np.linalg.solve(G_UU, RHS_U)
                    U_new = U_new_T.T
                except np.linalg.LinAlgError:
                    # Fallback to lstsq if singular
                    U_new_T, _, _, _ = np.linalg.lstsq(G_UU, RHS_U, rcond=None)
                    U_new = U_new_T.T
                
                # --- Update V ---
                # T[:, b, :] = sum_r V[b, r] (U[:, r] outer W[:, r])
                # H_UW[r] = (U[:, r] outer W[:, r]).reshape(-1) # (nm*pn,)
                H_UW = (U[:, :, None] * W[None, :, :]).reshape(target_rank, nm * pn)
                
                # RHS_V = H_UW.T @ T_flat_V.T
                # T_flat_V[b] = T[:, b, :].reshape(-1) -> shape (mp, nm*pn)
                T_flat_V = np.zeros((mp, nm * pn))
                # We can fill this efficiently or just use the tensor directly
                # T is (nm, mp, pn). We need T[:, b, :].reshape(-1) for each b.
                # This is equivalent to moving axis 1 to front and reshaping.
                T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                
                RHS_V = H_UW.T @ T_flat_V.T  # (target_rank, mp)
                G_VV = H_UW.T @ H_UW
                
                try:
                    V_new_T = np.linalg.solve(G_VV, RHS_V)
                    V_new = V_new_T.T
                except np.linalg.LinAlgError:
                    V_new_T, _, _, _ = np.linalg.lstsq(G_VV, RHS_V, rcond=None)
                    V_new = V_new_T.T
                
                # --- Update W ---
                # T[:, :, c] = sum_r W[c, r] (U[:, r] outer V[:, r])
                # H_UV[r] = (U[:, r] outer V[:, r]).reshape(-1) # (nm*mp,)
                H_UV = (U[:, :, None] * V[None, :, :]).reshape(target_rank, nm * mp)
                
                # RHS_W = H_UV.T @ T_flat_W.T
                # T_flat_W[c] = T[:, :, c].reshape(-1) -> shape (pn, nm*mp)
                T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                
                RHS_W = H_UV.T @ T_flat_W.T  # (target_rank, pn)
                G_WW = H_UV.T @ H_UV
                
                try:
                    W_new_T = np.linalg.solve(G_WW, RHS_W)
                    W_new = W_new_T.T
                except np.linalg.LinAlgError:
                    W_new_T, _, _, _ = np.linalg.lstsq(G_WW, RHS_W, rcond=None)
                    W_new = W_new_T.T
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence every few iterations
                if it % 5 == 0 or it == max_iter - 1:
                    err = compute_error(U, V, W)
                    if err < 1e-8:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                        return True
                    if err < prev_err * 0.95:
                        prev_err = err
                    elif it > 10 and err > prev_err * 1.05:
                        # Diverging or stuck, break early
                        break
            
            # Final check
            err = compute_error(U, V, W)
            if err < 1e-8:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                return True
        
        return False
    
    # Try ranks from schoolbook-1 down to a reasonable minimum
    # Lower bound is max(nm, mp, pn) = 28, but practically much higher.
    # Let's try down to 80 first, as 4x4x7 is likely in the 90s.
    candidate_ranks = list(range(rank_school - 1, 79, -1))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        # Increase restarts for lower ranks as they are harder
        n_restarts = 5 if rank > 100 else 10
        max_iter = 100 if rank > 100 else 200
        
        success = try_rank(rank, n_restarts=n_restarts, max_iter=max_iter)
        if success:
            break
    
    # If we found a better rank, convert to lists
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U_school.tolist()
        V_out = V_school.tolist()
        W_out = W_school.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
