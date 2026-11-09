    def svd_init(n, m, p, R):
        # Unfoldings
        # T1: (n*m, m*p*p*n) -> U_init from T1
        # T2: (m*p, n*m*p*n) -> V_init from T2
        # T3: (p*n, n*m*m*p) -> W_init from T3
        
        # We already have T1, T2, T3 as 2D arrays.
        # T1 is (n*m, (m*p)*(p*n))
        # We want U of shape (n*m, R)
        # Take top R left singular vectors of T1
        U, s1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        U_init = U[:, :R]
        
        V, s2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        V_init = V[:, :R]
        
        W, s3, Vt3 = np.linalg.svd(T3, full_matrices=False)
        W_init = W[:, :R]
        
        return U_init, V_init, W_init
