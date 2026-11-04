    def check_solution_vectorized(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        T = np.einsum('ibr,ckr,kir->ibck', U_3d, V_3d, W_3d)
        
        delta_bc = np.eye(m)
        E = np.broadcast_to(delta_bc[None, :, :, None], (n, m, m, p))
        
        residual = T - E
        return np.all(np.abs(residual) < tol)
