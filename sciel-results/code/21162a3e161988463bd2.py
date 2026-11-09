            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-10 * np.eye(n)
                D = np.linalg.cholesky(C)
