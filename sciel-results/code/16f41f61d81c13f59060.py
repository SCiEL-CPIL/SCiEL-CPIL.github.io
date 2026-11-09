"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts on BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget
        best_x = None
        best_f = np.inf

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Initial exploration: evaluate a few random points to get a good start
        rng = np.random.default_rng(42)

        # Try several starting points
        n_starts = 5
        starts = []
        for i in range(n_starts):
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            starts.append((f0, x0))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Sort by fitness, best first
        starts.sort(key=lambda t: t[0])

        # Use CMA-ES from the best start, with restarts if stagnation
        remaining = total_budget - len(starts)

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Run CMA-ES from best start
        x_best_local = starts[0][1].copy()
        f_best_local = starts[0][0]

        # Estimate sigma from spread of initial points
        if len(starts) > 1:
            spread = np.mean([np.linalg.norm(s[1] - x_best_local) for s in starts[:3]])
            sigma = max(0.5, min(2.0, spread / 2.0))
        else:
            sigma = 1.0

        # CMA-ES state
        x_mean = x_best_local.copy()
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_made = len(starts)

        # Track history for stagnation detection
        history = [(f_best_local, calls_made)]
        last_improvement_call = calls_made
        stagnation_threshold = 80  # no improvement in 80 calls -> restart

        for gen in range(1000):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Check stagnation
            if calls_made - last_improvement_call > stagnation_threshold and gen > 2:
                # Restart: try a different starting point
                improved = False
                for i in range(1, min(len(starts), 4)):
                    x_try = starts[i][1] + rng.normal(0, 0.5, dim)
                    x_try = clip(x_try)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < f_best_local:
                        x_mean = x_try.copy()
                        f_best_local = f_try
                        last_improvement_call = calls_made
                        improved = True
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                        break
                    if time.time() > deadline or calls_made >= total_budget:
                        break
                if not improved:
                    # Just reduce sigma and continue
                    sigma *= 0.5
                    last_improvement_call = calls_made

            # Generate population
            pop_x = []
            pop_f = []
            for i in range(lambda_pop):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Mahalanobis transform
                step = sigma * (B @ (D * z))
                y = clip(x_mean + step)
                fy = func(y)
                calls_made += 1
                pop_x.append(y)
                pop_f.append(fy)

            if len(pop_x) < mu:
                break

            # Sort by fitness
            sorted_idx = np.argsort(pop_f)
            pop_x_sorted = [pop_x[i] for i in sorted_idx[:mu]]
            pop_f_sorted = [pop_f[i] for i in sorted_idx[:mu]]

            # Update best
            if pop_f_sorted[0] < f_best_local:
                f_best_local = pop_f_sorted[0]
                x_best_local = pop_x_sorted[0].copy()
                last_improvement_call = calls_made
                if f_best_local < best_f:
                    best_f = f_best_local
                    best_x = x_best_local.copy()

            # Compute weighted mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * pop_x_sorted[i]

            # Update covariance matrix evolution path
            old_x_mean = x_mean.copy()
            x_mean = x_new_mean

            # Update sigma (CSA)
            z_mean = np.linalg.solve(np.diag(D), B.T @ (x_new_mean - old_x_mean)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            h_sigma = (np.linalg.norm(p_sigma) /
                       np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - last_improvement_call))) /
                       np.sqrt(1 - (1 - c_sigma) ** 2) if calls_made - last_improvement_call > 0 else 1)
            # Simplified h_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, gen))) < (1.4 + 1 / dim) * chi_n else 0

            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-10, min(sigma, 50.0))

            # Update CMA evolution path
            c_step = np.linalg.solve(np.diag(D), B.T @ (x_new_mean - old_x_mean)) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * c_step

            # Update B and D
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            C_update = np.outer(p_c, p_c)
            for i in range(mu):
                c_step_i = np.linalg.solve(np.diag(D), B.T @ (pop_x_sorted[i] - old_x_mean)) / sigma
                C_update += weights[i] * np.outer(c_step_i, c_step_i)

            # Use spectral decomposition to update B and D
            # Approximate: recompute from scratch periodically
            if gen % 10 == 0:
                # Re-estimate covariance from recent population
                if len(pop_x_sorted) >= dim + 1:
                    C_est = np.cov(np.array(pop_x_sorted).T, bias=True)
                    if not np.all(np.isfinite(C_est)):
                        C_est = np.eye(dim)
                    eigvals, eigvecs = np.linalg.eigh(C_est + 1e-6 * np.eye(dim))
                    eigvals = np.clip(eigvals, 1e-12, None)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                else:
                    B = np.eye(dim)
                    D = np.ones(dim)

        # Final polish: try small perturbations around best
        if best_x is not None and calls_made < total_budget:
            polish_calls = min(20, total_budget - calls_made)
            for i in range(polish_calls):
                if time.time() > deadline:
                    break
                for scale in [0.1, 0.05, 0.01]:
                    if calls_made >= total_budget:
                        break
                    x_try = clip(best_x + scale * rng.standard_normal(dim))
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
