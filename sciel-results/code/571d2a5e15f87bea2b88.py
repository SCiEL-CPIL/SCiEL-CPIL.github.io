<<<<<<< SEARCH
def _als_decomposition(T, rank, n=3, m=3, p=6, max_iters=100, tol=1e-8):
    """
    Alternating Least Squares for Tensor Decomposition.
    T: (p*n, n*m, m*p)
    Returns U, V, W or None if not converged.
    """
    dim_w, dim_u, dim_v = T.shape
    
    # Initialize using SVD of unfolded matrices
    # Unfold T along axis 0: (dim_w, dim_u * dim_v)
    T_u0 = T.reshape(dim_w, dim_u * dim_v)
    _, _, Vt0 = np.linalg.svd(T_u0, full_matrices=False)
    W = Vt0[:rank, :].T
    
    # Unfold T along axis 1: (dim_u, dim_w * dim_v)
    T_u1 = T.transpose(1, 0, 2).reshape(dim_u, dim_w * dim_v)
    _, _, Vt1 = np.linalg.svd(T_u1, full_matrices=False)
    U = Vt1[:rank, :].T
    
    # Unfold T along axis 2: (dim_v, dim_w * dim_u)
    T_u2 = T.transpose(2, 0, 1).reshape(dim_v, dim_w * dim_u)
    _, _, Vt2 = np.linalg.svd(T_u2, full_matrices=False)
    V = Vt2[:rank, :].T
    
    for it in range(max_iters):
        # Update U
        # T_decomp[b, a, c] = sum_r U[b, r] V[c, r] W[a, r]
        # For fixed r, U[:, r] = (W[:, r]^T V[:, r])^+ T_{:, :, r}^T ?
        # Let's solve for U[:, r] given V, W.
        # T[:, :, :] = sum_r U[:, r] (V[:, r] * W[:, r])^T ? No.
        # T[a, b, c] = sum_r U[b, r] V[c, r] W[a, r]
        # Fix r. We want U[:, r].
        # T[:, b, :] = sum_r' U[b, r'] V[:, r'] W[:, r']^T ?
        # Let's use matrix form.
        # T_{(1)} [b, a*c] = U[:, r] * (V[:, r] * W[:, r]) ?
        # Actually, T_{(1)} = U * (V \circ W)^T ? No.
        # T_{(1)}[b, (a,c)] = sum_r U[b, r] V[c, r] W[a, r]
        # Let M = V * W (element-wise product of V and W transposed? No)
        # M[r, (a,c)] = V[c, r] * W[a, r]
        # Then T_{(1)} = U * M
        # So U = T_{(1)} * M^+
        
        # Construct M for U update
        # M shape: (rank, dim_w * dim_v)
        # M[r, a * dim_v + c] = W[a, r] * V[c, r]
        M = np.zeros((rank, dim_w * dim_v))
        for r in range(rank):
            M[r, :] = (W[:, r] * V[:, r]).reshape(-1) # Wait, W is dim_w x rank, V is dim_v x rank
            # We need to align indices.
            # T[a, b, c]. a is W, b is U, c is V.
            # T_{(1)} is b x (a*c).
            # M[r, idx(a,c)] = W[a, r] * V[c, r]
            # We can compute this by:
            # W_expanded = W[:, :, None] * V[None, :, :] ? No.
            # W is (dim_w, rank). V is (dim_v, rank).
            # We want (dim_w * dim_v, rank) matrix N where N[a, c, r] = W[a, r] V[c, r].
            # N = np.einsum('ar,cr->acr', W, V)
            N = np.einsum('ar,cr->acr', W, V)
            M = N.reshape(dim_w * dim_v, rank).T
            
        # T_u1 is (dim_u, dim_w * dim_v)
        # U = T_u1 @ M^+
        try:
            U_new = T_u1 @ np.linalg.pinv(M)
        except:
            break
            
        # Update V
        # T_{(2)} is c x (a*b).
        # T[c, a, b] = sum_r V[c, r] U[b, r] W[a, r]
        # M_v[r, idx(a,b)] = W[a, r] * U[b, r]
        N = np.einsum('ar,br->abr', W, U)
        M = N.reshape(dim_w * dim_u, rank).T
        T_u2 = T.transpose(2, 0, 1).reshape(dim_v, dim_w * dim_u)
        try:
            V_new = T_u2 @ np.linalg.pinv(M)
        except:
            break
            
        # Update W
        # T_{(0)} is a x (b*c).
        # T[a, b, c] = sum_r W[a, r] U[b, r] V[c, r]
        # M_w[r, idx(b,c)] = U[b, r] * V[c, r]
        N = np.einsum('br,cr->bcr', U, V)
        M = N.reshape(dim_u * dim_v, rank).T
        T_u0 = T.reshape(dim_w, dim_u * dim_v)
        try:
            W_new = T_u0 @ np.linalg.pinv(M)
        except:
            break
            
        U, V, W = U_new, V_new, W_new
        
        # Check error
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.linalg.norm(T_decomp - T)
        if err < tol:
            return U, V, W
            
    return None
=======
def _als_decomposition(T, rank, n=3, m=3, p=6, max_iters=100, tol=1e-8, deadline=None):
    """
    Alternating Least Squares for Tensor Decomposition.
    T: (p*n, n*m, m*p)
    Returns U, V, W or None if not converged.
    """
    dim_w, dim_u, dim_v = T.shape
    
    # Initialize using SVD of unfolded matrices
    T_u0 = T.reshape(dim_w, dim_u * dim_v)
    _, _, Vt0 = np.linalg.svd(T_u0, full_matrices=False)
    W = Vt0[:rank, :].T
    
    T_u1 = T.transpose(1, 0, 2).reshape(dim_u, dim_w * dim_v)
    _, _, Vt1 = np.linalg.svd(T_u1, full_matrices=False)
    U = Vt1[:rank, :].T
    
    T_u2 = T.transpose(2, 0, 1).reshape(dim_v, dim_w * dim_u)
    _, _, Vt2 = np.linalg.svd(T_u2, full_matrices=False)
    V = Vt2[:rank, :].T
    
    # Precompute unfolded matrices
    T_u0_mat = T.reshape(dim_w, dim_u * dim_v)
    T_u1_mat = T.transpose(1, 0, 2).reshape(dim_u, dim_w * dim_v)
    T_u2_mat = T.transpose(2, 0, 1).reshape(dim_v, dim_w * dim_u)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            return None
            
        # Update U
        # T[a,b,c] = sum_r W[a,r] U[b,r] V[c,r]
        # T_{(1)}[b, (a,c)] = sum_r U[b,r] (W[a,r] V[c,r])
        # M_U[r, (a,c)] = W[a,r] * V[c,r]
        # M_U is (rank, dim_w * dim_v)
        # Construct M_U
        # W is (dim_w, rank), V is (dim_v, rank)
        # We need outer product of W[:,r] and V[:,r] for each r
        # This is the Khatri-Rao product of W and V, transposed?
        # Let's use einsum to get (dim_w, dim_v, rank) then reshape
        # K_r = W[:, r] * V[:, r] (element-wise product of vectors? No, outer product)
        # Actually, for fixed r, the vector of size dim_w*dim_v is W[:,r] \otimes V[:,r]
        # M_U[r, :] = (W[:, r] * V[:, r]) is wrong. It should be np.kron(W[:, r], V[:, r])
        # Let's compute M_U for all r at once.
        # M_U[r, a*dim_v + c] = W[a, r] * V[c, r]
        # This is equivalent to:
        # W_exp = W[:, :, None] # (dim_w, rank, 1)
        # V_exp = V[None, :, :] # (1, dim_v, rank)
        # Product = W_exp * V_exp # (dim_w, dim_v, rank)
        # M_U = Product.reshape(dim_w * dim_v, rank).T # (rank, dim_w * dim_v)
        
        M_U = np.einsum('ar,cr->acr', W, V).reshape(dim_w * dim_v, rank).T
        try:
            U_new = T_u1_mat @ np.linalg.pinv(M_U)
        except:
            return None
            
        # Update V
        # T_{(2)}[c, (a,b)] = sum_r V[c,r] (W[a,r] U[b,r])
        # M_V[r, (a,b)] = W[a,r] * U[b,r]
        M_V = np.einsum('ar,br->abr', W, U).reshape(dim_w * dim_u, rank).T
        try:
            V_new = T_u2_mat @ np.linalg.pinv(M_V)
        except:
            return None
            
        # Update W
        # T_{(0)}[a, (b,c)] = sum_r W[a,r] (U[b,r] V[c,r])
        # M_W[r, (b,c)] = U[b,r] * V[c,r]
        M_W = np.einsum('br,cr->bcr', U, V).reshape(dim_u * dim_v, rank).T
        try:
            W_new = T_u0_mat @ np.linalg.pinv(M_W)
        except:
            return None
            
        U, V, W = U_new, V_new, W_new
        
        # Check error
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.linalg.norm(T_decomp - T)
        if err < tol:
            return U, V, W
            
    return None
>>>>>>> REPLACE

