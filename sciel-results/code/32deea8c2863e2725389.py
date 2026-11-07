        def gp_predict(X_test, X_train, Y_train, l, sf, sn):
            K = matern52_kernel(X_train, X_train, l) + sn * np.eye(len(X_train))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X_train)))
                
            K_s = matern52_kernel(X_test, X_train, l)
            
            # Solve L v = K_s.T
            v = np.linalg.solve(L, K_s.T)
            # Solve L.T alpha = Y_train
            alpha = np.linalg.solve(L.T, Y_train)
            
            mean = K_s @ alpha
            var = sf**2 - np.sum(K_s * v, axis=1)
            var = np.maximum(var, 1e-8)
            return mean, np.sqrt(var)
