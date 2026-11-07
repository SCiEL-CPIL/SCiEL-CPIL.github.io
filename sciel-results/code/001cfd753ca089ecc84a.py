import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        rng = np.random.default_rng(42)

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid initialization to find a promising basin
        # For 5D, use 2 points per dim = 32 evaluations
        n_init = min(32, budget)
        if n_init >= 32:
            grid_vals = np.linspace(-4.0, 4.0, 2)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            for m in range(2):
                                if calls_made >= budget or time.time() > deadline:
                                    break
                                x = np.array([grid_vals[i], grid_vals[j], grid_vals[k], grid_vals[l], grid_vals[m]])
                                counted_func(x)
        else:
            for _ in range(n_init):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-4.5, 4.5, dim)
                counted_func(x)

        # Phase 2: CMA-ES optimization
        n = dim
        lam = 4 + int(3 * np.log(n))
        if lam < 4:
            lam = 4
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        cs = (mueff + 1.3) / (n + mueff + 3)
        ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # Initialize CMA-ES from the best point found in Phase 1
        x_mean = best_x.copy()
        sigma = 1.5
        C = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Stagnation handling
        last_best_f = best_f
        stagnation_count = 0
        max_stagnation = 30

        # Coordinate refinement step sizes
        base_steps = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
        step_idx = 0
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # Golden-section ratio

        # CMA-ES main loop
        gen = 0
        while calls_made < budget and time.time() < deadline:
            if budget - calls_made < lam:
                break

            # Generate population
            Z = rng.standard_normal((lam, n))
            Y = Z @ C_sqrt_inv.T
            X = x_mean + sigma * Y

            # Evaluate population
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                F[i] = counted_func(X[i])

            # Check if we completed the generation
            if calls_made < lam or F[-1] == np.inf:
                break

            gen += 1

            # Update CMA-ES
            idx = np.argsort(F)
            top_mu_idx = idx[:mu]
            top_mu_X = X[top_mu_idx]

            # Weighted mean
            x_new_mean = np.sum(w[:, np.newaxis] * top_mu_X, axis=0)

            # Update p_sigma
            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * diff

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Update p_c
            p_c = (1 - cs) * p_c + cs * np.sqrt(mueff) * diff

            # h_sigma
            if gen > 0:
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 * gen))
                if np.linalg.norm(p_sigma) / denom < (1.4 + 2 / (n + 1)) * chi_n:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
            else:
                h_sigma = 1.0

            # Update C using rank-mu update
            cc = 2 * mueff * cs * (2 - cs)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = mueff * (c1 - c1)  # Simplified; using standard rank-mu
            # More standard: c1 = 2/((n+1.3)^2 + mueff), cmu = mueff * c1
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = mueff * c1

            C = (1 - cc - c1 - cmu) * C + cc * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (top_mu_X[i] - x_mean) / sigma
                y_i = y_i - p_c
                C += cmu * w[i] * np.outer(y_i, y_i)

            # Ensure C is symmetric
            C = (C + C.T) / 2.0

            # Update C_sqrt via eigenvalue decomposition (for small n)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update x_mean
            x_mean = x_new_mean

            # Stagnation check
            if best_f < last_best_f - 1e-8:
                last_best_f = best_f
                stagnation_count = 0
            else:
                stagnation_count += lam

            # If stagnating, do coordinate refinement
            if stagnation_count > max_stagnation:
                # Try coordinate-wise refinement around best_x
                step = base_steps[min(step_idx, len(base_steps) - 1)]
                for coord in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    # Try moving in + and - directions
                    x_plus = best_x.copy()
                    x_plus[coord] = best_x[coord] + step
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    f_plus = counted_func(x_plus)

                    if calls_made >= budget or time.time() > deadline:
                        break

                    x_minus = best_x.copy()
                    x_minus[coord] = best_x[coord] - step
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    f_minus = counted_func(x_minus)

                    if calls_made >= budget or time.time() > deadline:
                        break

                    # If we found improvement, refine further
                    if f_plus < best_f:
                        for sub in range(2):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            new_step = step * (0.5 ** (sub + 1))
                            x_new = best_x.copy()
                            x_new[coord] = best_x[coord] + new_step
                            x_new = np.clip(x_new, -5.0, 5.0)
                            counted_func(x_new)
                    elif f_minus < best_f:
                        for sub in range(2):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            new_step = step * (0.5 ** (sub + 1))
                            x_new = best_x.copy()
                            x_new[coord] = best_x[coord] - new_step
                            x_new = np.clip(x_new, -5.0, 5.0)
                            counted_func(x_new)

                # Shrink step for next refinement
                if step_idx < len(base_steps) - 1:
                    step_idx += 1

                # Restart CMA-ES with fresh state from best point
                x_mean = best_x.copy()
                sigma = 0.5
                C = np.eye(n)
                C_sqrt = np.eye(n)
                C_sqrt_inv = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                stagnation_count = 0
                last_best_f = best_f

        return best_x, best_f
