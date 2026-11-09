"""Multi-start CMA-ES optimizer for BBOB f21 with structured initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES optimizer with structured initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Structured initialization
        n_init = 30
        init_points = []

        # Center
        init_points.append(np.zeros(dim))

        # Axis-aligned points
        for d in range(dim):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                p = np.zeros(dim)
                p[d] = val
                init_points.append(p)

        # Random points to fill up to n_init
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.0, 4.0, dim))

        init_points = np.array(init_points[:n_init])

        init_results = []
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(init_points[i])
            init_results.append((init_points[i].copy(), f))

        init_results.sort(key=lambda x: x[1])

        # Phase 2: CMA-ES on top candidates
        K = min(3, len(init_results))
        if K == 0:
            return best_x, best_f

        top_starts = init_results[:K]
        remaining_budget = budget - calls_made
        budget_per_start = max(50, remaining_budget // K)

        for x0, f0 in top_starts:
            if calls_made >= budget or time.time() > deadline:
                break

            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(dim))  # ~11 for dim=5
            max_iter = min(budget_per_start // pop_size, 30)
            sigma = 1.0
            mean = x0.copy()
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            step_size = 1.0

            # Adaptation rates
            cmu = 2 / ((dim + 1)**2 + 2)
            c_sigma = (mu_eff + 0.5) / (dim + mu_eff + 5)
            mu_eff = 0.5 * pop_size
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(cmu, 2 * (1 - c_1) / ((dim + 2)**2 + mu_eff))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            d_sigma = 1 + 2 * max(0, (c_sigma - 1) / (1 - c_sigma)) + c_sigma

            for it in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Generate population
                ys = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (C_sqrt @ z)
                    ys.append(y)
                ys = np.array(ys)

                # Evaluate
                fs = []
                for y in ys:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    fy = evaluate(y)
                    fs.append(fy)

                if len(fs) < pop_size:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys_sorted = ys[idx]
                fs_sorted = fs[idx]

                # Update mean (weighted average of top half)
                weights = np.zeros(pop_size)
                weights[:int(pop_size / 2)] = np.log(int(pop_size / 2) + 0.5) - np.log(1 + np.arange(int(pop_size / 2)))
                weights /= weights.sum()

                old_mean = mean.copy()
                mean = np.sum(weights[:, None] * ys_sorted, axis=0)

                # Update path
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt.T @ (mean - old_mean))
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (C_sqrt.T @ (mean - old_mean))

                # Update covariance
                delta_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) - 1) * c_sigma * d_sigma
                sigma = sigma * np.exp(c_sigma * delta_sigma)
                sigma = max(0.001, min(sigma, 10.0))

                outer = np.sum(weights[:, None] * np.outer(ys_sorted - old_mean, ys_sorted - old_mean), axis=0)
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(it + 1)))**2 < (1 + 4 / dim) * (1 - c_sigma) else 0.0
                C = (1 - c_1 - c_mu + c_mu * h_sigma) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * C) + c_mu * outer

                # Re-orthogonalize C
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

                # Check convergence
                if sigma < 0.001 or np.all(np.abs(np.diff(fs_sorted[:3])) < 1e-8):
                    break

        # Phase 3: Fine-tuning with small random steps around best
        if calls_made < budget and time.time() < deadline and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.05
            for _ in range(30):
                if calls_made >= budget or time.time() > deadline:
                    break
                y = x_polish + sigma_p * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p = max(0.01, sigma_p * 0.9)
                else:
                    sigma_p = max(0.005, sigma_p * 0.8)

        return best_x, best_f
# EVOLVE-BLOCK-END
