<<<<<<< SEARCH
            sigma *= np.exp(c_sigma / d_sigma if (d_sigma := np.sqrt((1 - (1 - c_sigma) ** (2 * (call_count - lambda_))) * (1 - (1 - c_c) ** (2 * (call_count - lambda_))) * (1 + 2 / (dim + 1)))) else 1.0) * (np.linalg.norm(C_sqrt.T @ p_sigma) / d_norm - 1))
=======
            d_sigma = np.sqrt((1 - (1 - c_sigma) ** (2 * (call_count - lambda_))) * (1 - (1 - c_c) ** (2 * (call_count - lambda_))) * (1 + 2 / (dim + 1)))
            sigma *= np.exp(c_sigma / d_sigma) * (np.linalg.norm(C_sqrt.T @ p_sigma) / d_norm - 1)
>>>>>>> REPLACE
