                # Update C_sqrt
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    sigma *= 0.5
