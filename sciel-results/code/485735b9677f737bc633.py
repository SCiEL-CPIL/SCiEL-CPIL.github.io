                    try:
                        AtA = A.T @ A
                        Atb = A.T @ b
                        # Add small ridge for stability
                        AtA[np.diag_indices_from(AtA)] += 1e-10
                        U[i*m+j, :] = np.linalg.solve(AtA, Atb)
                    except:
                        pass
