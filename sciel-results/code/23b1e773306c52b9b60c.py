    def try_find_rank_16_block():
        """Try to find a rank 16 decomposition for 3x3x2."""
        # Vectorized residual calculation
        def residuals(x):
            U = x[:144].reshape(9, 16)
            V = x[144:240].reshape(6, 16)
            W = x[240:336].reshape(6, 16)
            
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # Use einsum for speed
            # U: (9, 16) -> (3, 3, 16)
            # V: (6, 16) -> (3, 2, 16)
            # W: (6, 16) -> (2, 3, 16)
            U_3d = U.reshape(3, 3, 16)
            V_3d = V.reshape(3, 2, 16)
            W_3d = W.reshape(2, 3, 16)
            
            # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
            # This is a bit tricky with einsum directly because of the index permutations.
            # T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            return (T - 1.0).flatten()

        # Try multiple random starts
        np.random.seed(42)
        for _ in range(50):
            x0 = np.random.randn(336) * 2
            # Ensure some structure?
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if np.max(np.abs(result.fun)) < 1e-6:
                    x_opt = result.x
                    U_opt = x_opt[:144].reshape(9, 16)
                    V_opt = x_opt[144:240].reshape(6, 16)
                    W_opt = x_opt[240:336].reshape(6, 16)
                    
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify rounded
                    T_check = np.einsum('ijc,jkc,kic->ijk', 
                                        U_rounded.reshape(3,3,16), 
                                        V_rounded.reshape(3,2,16), 
                                        W_rounded.reshape(2,3,16))
                    if np.max(np.abs(T_check - 1.0)) < 1e-10:
                        return U_rounded, V_rounded, W_rounded
            except Exception:
                pass
        return None
