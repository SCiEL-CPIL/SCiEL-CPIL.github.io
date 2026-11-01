    def verify_decomposition(U, V, W, rank, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for trial in range(3):
            A_rand = np.random.randn(n, m)
            B_rand = np.random.randn(m, p)
            C_expected = A_rand @ B_rand
            
            C_check = np.zeros((n, p))
            A_flat = A_rand.flatten()
            B_flat = B_rand.flatten()
            
            for r in range(rank):
                scalar = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
                w_vec = W_arr[:, r]
                for k in range(p):
                    for i in range(n):
                        C_check[i, k] += w_vec[k * n + i] * scalar
            
            if not np.allclose(C_check, C_expected, atol=1e-10):
                return False
        return True
