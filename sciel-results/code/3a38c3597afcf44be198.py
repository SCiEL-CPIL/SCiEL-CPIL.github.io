"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal Natural Gradient Descent optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 275  # safety margin vs 330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(42)

        # Population size for natural gradient estimation
        lambda_ = 24
        mu_ = 12

        # Weights for rank-mu update
        w = np.zeros(mu_)
        for i in range(mu_):
            w[i] = np.log(mu_ + 0.5) - np.log(i + 1)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1) + c_sigma
        c_diag = 2 / (dim ** 2 + 6)  # learning rate for diagonal adaptation

        # State
        x_mean = np.zeros(dim)
        sigma = 2.0  # global step size
        diag = np.ones(dim)  # per-dimension scale factors (log-space for stability)
        log_diag = np.zeros(dim)
        p_sigma = np.zeros(dim)
        p_diag = np.zeros(dim)  # evolution path for diagonal adaptation

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        call_count = 1

        # Chi_n constant
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while call_count < budget:
            if timed_out():
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if call_count >= budget or timed_out():
                    break

                z = rng.standard_normal(dim)
                # Scale by global sigma and per-dimension diag
                step = sigma * np.exp(log_diag) * z
                y = x_mean + step
                y = np.clip(y, -5, 5)
                f_y = func(y)
                call_count += 1
                offspring.append((y, f_y, z))

                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()

            if len(offspring) < mu_:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu_]

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * y for i, (y, f, z) in enumerate(selected)], axis=0)

            # Update sigma via evolution path
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                      np.sum([w[i] * z for i, (y, f, z) in enumerate(selected)], axis=0)
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update diagonal adaptation (rank-1 + rank-mu for diag)
            # The idea: adapt log_diag based on which directions helped
            # Use the weighted mean of z^2 for the selected points as a signal
            # This is a simplified natural gradient for the diagonal

            # Compute the weighted sum of outer products of z for selected points
            # For diagonal: we want to increase diag in directions where progress was made
            # Simple approach: use the mean squared z of selected vs overall population

            # Rank-1 update for diagonal path
            delta_log_diag = np.zeros(dim)
            for i in range(mu_):
                z_i = selected[i][2]
                delta_log_diag += w[i] * z_i * z_i  # second moment signal

            # Normalize: if selected points have smaller z^2 in some direction,
            # it means the search in that direction was more effective (smaller steps worked)
            # Actually, we want to INCREASE diag where the function is flatter
            # A better signal: compare the selected z^2 to the expected chi^2 value (1 for standard normal)

            # Use the difference from 1 (expected value of z^2 for standard normal)
            # If z^2 < 1 on average for selected, that direction was "easier" -> increase diag
            # If z^2 > 1, that direction was "harder" -> decrease diag

            signal = 1.0 - delta_log_diag  # positive means increase diag

            # Apply with learning rate
            p_diag = (1 - c_diag) * p_diag + c_diag * np.sqrt(mu_eff) * signal
            log_diag += c_diag * np.sqrt(mu_eff) * signal
            log_diag = np.clip(log_diag, -5, 5)  # prevent extreme scaling

            # Ensure log_diag stays reasonable
            log_diag = np.clip(log_diag, -3, 3)

        return best_x, best_f
# EVOLVE-BLOCK-END
