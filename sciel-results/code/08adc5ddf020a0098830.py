<<<<<<< SEARCH
            # Update sigma
            sigma = sigma * np.exp(c_sigma / ds * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update mean
            mean = new_mean

            # Track best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()

        return best_x, best_f, evals
=======
            # Update sigma
            sigma = sigma * np.exp(c_sigma / ds * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update mean
            mean = new_mean

            # Track best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()

        return best_x, best_f, evals
>>>>>>> REPLACE
