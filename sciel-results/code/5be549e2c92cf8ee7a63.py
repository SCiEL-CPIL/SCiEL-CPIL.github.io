<<<<<<< SEARCH
    best_rank = 49
    best_U = U49
    best_V = V49
    best_W = W49

    # Try to find a linear dependency: T_elim = sum_{r != elim} c_r * T_r
    # This means the 49 vectors are linearly dependent.
    # For each elim, solve the underdetermined system.
    
    for elim in range(49):
        if time.time() - t0 > deadline - 30:
            break
        
        # Get the 48 other tensors
        mask = [r for r in range(49) if r != elim]
        M = T_vecs[mask]  # shape (48, 4096)
        target = T_vecs[elim]
        
        # Solve M.T @ c = target (least squares)
        # M.T is (4096, 48), we want c such that M.T @ c ≈ target
        try:
            c, residuals, rank_M, sv = np.linalg.lstsq(M.T, target, rcond=1e-12)
            recon = M.T @ c
            err = np.linalg.norm(recon - target) / max(1.0, np.linalg.norm(target))
            
            if err < 1e-8:
                # T_elim = sum_{idx} c[idx] * T_{mask[idx]}
                # T_full = sum_{r != elim} T_r + T_elim = sum_{idx} (1 + c[idx]) * T_{mask[idx]}
                # For each term, scale: (1+c[idx]) * T_r = a*U_r ⊗ b*V_r ⊗ d*W_r
                # where a*b*d = 1+c[idx]. We can distribute: a = b = d = (1+c[idx])^(1/3)
                # But if 1+c[idx] < 0, we need to handle the sign.
                # For now, assume all (1+c[idx]) > 0.
                
                U_new = np.zeros((16, 48), dtype=np.float64)
                V_new = np.zeros((16, 48), dtype=np.float64)
                W_new = np.zeros((16, 48), dtype=np.float64)
                
                valid = True
                for idx, r in enumerate(mask):
                    scale_total = 1.0 + c[idx]
                    if scale_total < -1e-12:
                        # Negative scale - can't take real cube root easily
                        # Try: absorb sign into one factor
                        # scale_total * U ⊗ V ⊗ W = (-1) * (-scale_total) * U ⊗ V ⊗ W
                        # = (-U) ⊗ (V) ⊗ (W) * (-scale_total)
                        # Still need cube root of negative... use: 
                        # a = -abs(scale_total)^(1/3), b = abs(scale_total)^(1/3), d = abs(scale_total)^(1/3)
                        # a*b*d = -abs(scale_total) = scale_total ✓
                        a = -abs(scale_total) ** (1.0/3.0)
                        b = abs(scale_total) ** (1.0/3.0)
                        d = abs(scale_total) ** (1.0/3.0)
                    else:
                        a = scale_total ** (1.0/3.0)
                        b = a
                        d = a
                    
                    U_new[:, idx] = a * U49[:, r]
                    V_new[:, idx] = b * V49[:, r]
                    W_new[:, idx] = d * W49[:, r]
                
                # Verify
                err_verify = 0.0
                for i in range(4):
                    for k in range(4):
                        for j in range(4):
                            s = 0.0
                            for s_idx in range(48):
                                s += U_new[i*4+k, s_idx] * V_new[k*4+j, s_idx] * W_new[j*4+i, s_idx]
                            err_verify += abs(s - 1.0)
                
                if err_verify < 1e-6:
                    best_rank = 48
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    break
        except:
            pass

    if best_rank == 48:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
=======
    # Try to find a linear dependency: T_elim = sum_{r != elim} c_r * T_r
    # This means the 49 vectors are linearly dependent.
    # For each elim, solve the underdetermined system.
    
    # Use a more efficient approach: check if the 49 vectors span a space
    # of dimension < 49 by computing the rank of the matrix whose rows are T_vecs
    try:
        # Compute SVD to check rank
        U_svd, svals, Vt = np.linalg.svd(T_vecs, full_matrices=False)
        # Check if there's a significant null space
        rank_T = np.sum(svals > 1e-10 * svals[0])
        
        if rank_T < 49:
            # There's a linear dependency. Find it via the null space.
            # The null space of T_vecs (as a 49x4096 matrix) gives dependencies.
            # Actually, we want: sum_r alpha_r * T_vecs[r] = 0
            # This is: T_vecs @ alpha = 0, where alpha is (49,)
            # The null space of T_vecs.T (4096x49) gives this.
            # But since rank < 49, the last singular vector of T_vecs.T is in the null space.
            null_vec = Vt[-1, :]  # This is the right singular vector corresponding to smallest s
            # Actually Vt is (49, min(49,4096)) = (49, 49) if full_matrices=False and 49<4096
            # Wait, T_vecs is (49, 4096). SVD: T_vecs = U_svd @ diag(svals) @ Vt
            # U_svd: (49, 49), svals: (49,), Vt: (49, 4096)
            # Null space of T_vecs: vectors x such that T_vecs @ x = 0
            # x is in R^4096, so it's the null space of the 49x4096 matrix
            # That's given by the columns of Vt corresponding to zero singular values
            # But we want the dependency among the rows: sum_r alpha_r * row_r = 0
            # That's: alpha^T @ T_vecs = 0, i.e., T_vecs.T @ alpha = 0
            # alpha is in R^49, null space of T_vecs.T (4096x49)
            # Since rank < 49, there exists alpha != 0 with T_vecs.T @ alpha = 0
            
            # Compute null space of T_vecs.T
            # T_vecs.T is (4096, 49)
            U_t, svals_t, Vt_t = np.linalg.svd(T_vecs.T, full_matrices=False)
            # Vt_t is (49, 49)
            # The last row of Vt_t corresponds to the null space direction
            alpha = Vt_t[-1, :]
            
            # Find which alpha components are significant
            alpha_norm = np.linalg.norm(alpha)
            if alpha_norm > 1e-10:
                alpha = alpha / alpha_norm
                
                # Find the largest |alpha_r| to use as elim
                elim = int(np.argmax(np.abs(alpha)))
                mask = [r for r in range(49) if r != elim]
                
                # T_elim = -sum_{r != elim} (alpha_r / alpha_elim) * T_r
                c = -alpha[mask] / alpha[elim]
                
                # T_full = sum_{r != elim} T_r + T_elim = sum_{idx} (1 + c[idx]) * T_{mask[idx]}
                U_new = np.zeros((16, 48), dtype=np.float64)
                V_new = np.zeros((16, 48), dtype=np.float64)
                W_new = np.zeros((16, 48), dtype=np.float64)
                
                for idx, r in enumerate(mask):
                    scale_total = 1.0 + c[idx]
                    if abs(scale_total) < 1e-12:
                        # This term vanishes; leave as zeros
                        continue
                    if scale_total < 0:
                        a = -abs(scale_total) ** (1.0/3.0)
                        b = abs(scale_total) ** (1.0/3.0)
                        d = abs(scale_total) ** (1.0/3.0)
                    else:
                        a = scale_total ** (1.0/3.0)
                        b = a
                        d = a
                    
                    U_new[:, idx] = a * U49[:, r]
                    V_new[:, idx] = b * V49[:, r]
                    W_new[:, idx] = d * W49[:, r]
                
                # Verify
                U4n = U_new.reshape(4, 4, 48)
                V4n = V_new.reshape(4, 4, 48)
                W4n = W_new.reshape(4, 4, 48)
                T_check = np.einsum('ikr,kjr,jir->ikj', U4n, V4n, W4n)
                err_verify = np.abs(T_check - 1.0).sum()
                
                if err_verify < 1e-6:
                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    except:
        pass

    # Fallback: return rank-49
    return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
>>>>>>> REPLACE
