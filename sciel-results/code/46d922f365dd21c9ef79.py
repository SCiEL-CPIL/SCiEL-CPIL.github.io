            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.linalg.cholesky(C + 1e-6 * np.eye(d))
