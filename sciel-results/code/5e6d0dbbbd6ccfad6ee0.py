            Y = (X_sorted[:mu] - x_m) / sigma
            # Weighted sum of outer products
            # Y is (mu, dim)
            # We want sum_i w_i * y_i * y_i^T
            C_update = np.sum(weights[:mu, None, None] * Y[:, :, None] * Y[:, None, :], axis=0)
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * C_update
