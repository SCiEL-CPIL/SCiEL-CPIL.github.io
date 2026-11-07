"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer for Schaffer's F7 (ill-conditioned, axis-aligned):
    1. Coarse random sampling to find a good basin.
    2. CMA-ES for global/local refinement with restarts on stagnation.
    3. Coordinate-wise line search to exploit the axis-aligned ill-conditioned structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 290.0  # 10s safety margin before 300s wall-clock limit

        def time_ok():
            return time.time() < DEADLINE

        def safe_call(x):
            """Call func with clipping, return (x_clipped, f_value)."""
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                fval = float(func(x_clipped))
            except Exception:
                fval = np.inf
            if not np.isfinite(fval):
                fval = np.inf
            return x_clipped, fval

        # Stage 1: Coarse random sampling
        rng = np.random.default_rng(42)
        n_samples = 40
        best_x = None
        best_f = np.inf
        samples = []
        for _ in range(n_samples):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            x_c, f = safe_call(x)
            samples.append((x_c, f))
            if f < best_f:
                best_f = f
                best_x = x_c.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = np.inf

        # Stage 2: CMA-ES with restarts
        # We implement a lightweight CMA-ES here.
        # Parameters: population size, sigma, etc.
        def cmaes(x0, f0, budget_remaining, deadline_check):
            """Run CMA-ES starting from x0 with initial f0."""
            dim = self.dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            # Initial sigma: adapt based on box size
            sigma = 1.0
            # CMA-ES state
            x_mean = x0.copy()
            f_mean = f0
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            # Step size
            step_size = sigma
            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            # Adaptation rates
            c_sigma = (lam + 2) / (dim + 4)
            c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            mu = lam // 2  # number of weights
            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / (w @ w)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim * dim))

            best_x_local = x0.copy()
            best_f_local = f0
            calls_made = 0
            stagnation_count = 0
            max_stagnation = 50  # restart after this many iterations without improvement

            for iteration in range(budget_remaining // lam + 1):
                if not deadline_check() or calls_made >= budget_remaining:
                    break

                # Generate population
                B = rng.multivariate_normal(np.zeros(dim), C, size=lam)
                # Normalize to unit length
                B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
                # Sort by norm for recombination
                norms = np.linalg.norm(B, axis=1)
                idx = np.argsort(norms)
                B = B[idx]
                # Take top mu
                B = B[:mu]

                # Generate offspring
                y = []
                f_y = []
                for i in range(mu):
                    y_i = x_mean + step_size * (B[i] @ C_sqrt)
                    y_i = np.clip(y_i, -5, 5)
                    y.append(y_i)
                    calls_made += 1
                    if calls_made >= budget_remaining or not deadline_check():
                        break
                    _, f_y_i = safe_call(y_i)
                    f_y.append(f_y_i)

                if len(f_y) < mu:
                    break

                # Sort by fitness
                order = np.argsort(f_y)
                y = [y[i] for i in order]
                f_y = [f_y[i] for i in order]

                # Update best
                if f_y[0] < best_f_local:
                    best_f_local = f_y[0]
                    best_x_local = y[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Update mean
                x_mean_new = sum(w[i] * y[i] for i in range(mu))
                delta_mean = (x_mean_new - x_mean) / step_size

                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_mean
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) / chi_n < 1.4 + 2 / (dim + 1)
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean

                # Update covariance
                C = (1 + c_c * (2 - c_c) - c_sigma) * C
                rank_one = np.outer(delta_mean, delta_mean)
                rank_mu = sum(w[i] * np.outer(y[i] - x_mean, y[i] - x_mean) for i in range(mu))
                C += c_sigma * rank_one + c_c * (2 - c_c) * rank_mu

                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)

                x_mean = x_mean_new

                # Adapt step size
                step_size *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                step_size = np.clip(step_size, 1e-10, 10.0)

                # Restart on stagnation
                if stagnation_count > max_stagnation:
                    break

            return best_x_local, best_f_local, calls_made

        # Run CMA-ES from the best sample
        remaining = budget - n_samples
        if remaining > 0 and time_ok():
            best_x, best_f, _ = cmaes(best_x, best_f, remaining, time_ok)

        # Stage 3: Coordinate-wise line search (exploit axis-aligned structure)
        # For each coordinate, do a fine-grained search around the current best
        if time_ok() and best_f < np.inf:
            # Use remaining budget for coordinate search
            calls_used = n_samples
            # We need to track calls properly; let's just do a few coordinate refinements
            # Simple approach: for each coordinate, try a few perturbations
            for coord in range(dim):
                if not time_ok():
                    break
                # Try a few step sizes
                for step in [0.1, 0.05, 0.02, 0.01]:
                    if not time_ok():
                        break
                    # Try + and -
                    for direction in [1, -1]:
                        if not time_ok():
                            break
                        x_new = best_x.copy()
                        x_new[coord] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        _, f_new = safe_call(x_new)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
