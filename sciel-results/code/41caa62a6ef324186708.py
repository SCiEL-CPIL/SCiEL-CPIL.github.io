def als_solve(U, V, W, target_rank, rng, max_iter=50):
    R = target_rank
    U_cur = U[:, :R].copy()
    V_cur = V[:, :R].copy()
    W_cur = W[:, :R].copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            return None
        
        # Build KR_VW (392, R)
        KR_VW = np.zeros((m*p*p*n, R))
        for r in range(R):
            KR_VW[:, r] = np.kron(V_cur[:, r], W_cur[:, r])
        
        # U_new: solve T_flat_U = U_new @ KR_VW^T
        # => U_new = T_flat_U @ (KR_VW^T)^+ 
        # Use: U_new = T_flat_U @ KR_VW @ inv(KR_VW^T @ KR_VW)
        try:
            KR_VW_T_KR_VW = KR_VW.T @ KR_VW  # (R, R)
            U_new = T_flat_U @ KR_VW @ np.linalg.inv(KR_VW_T_KR_VW)
        except:
            U_new = T_flat_U @ np.linalg.pinv(KR_VW.T)
        
        # Build KR_WU (112, R)
        KR_WU = np.zeros((p*n*n*m, R))
        for r in range(R):
            KR_WU[:, r] = np.kron(W_cur[:, r], U_new[:, r])
        
        try:
            KR_WU_T_KR_WU = KR_WU.T @ KR_WU
            V_new = T_flat_V @ KR_WU @ np.linalg.inv(KR_WU_T_KR_WU)
        except:
            V_new = T_flat_V @ np.linalg.pinv(KR_WU.T)
        
        # Build KR_UV (224, R)
        KR_UV = np.zeros((n*m*m*p, R))
        for r in range(R):
            KR_UV[:, r] = np.kron(U_new[:, r], V_new[:, r])
        
        try:
            KR_UV_T_KR_UV = KR_UV.T @ KR_UV
            W_new = T_flat_W @ KR_UV @ np.linalg.inv(KR_UV_T_KR_UV)
        except:
            W_new = T_flat_W @ np.linalg.pinv(KR_UV.T)
        
        change = (np.linalg.norm(U_new - U_cur) + 
                 np.linalg.norm(V_new - V_cur) + 
                 np.linalg.norm(W_new - W_cur))
        U_cur, V_cur, W_cur = U_new, V_new, W_new
        if change < 1e-12:
            break
    
    return U_cur, V_cur, W_cur
