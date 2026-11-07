"""Optimizer using a model-and-optimize approach for BBOB f13 (Sharp Ridge).

Strategy:
1. Initial random sampling to get a rough estimate of the optimum.
2. Use a CMA-ES variant that adapts its covariance matrix to align with the ridge.
3. Use a budget-splitting scheme with multiple restarts.
4. Each restart uses a smaller population and adapts quickly.
5. Local refinement with smaller steps at the end.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using a multi-start CMA-ES approach for Sharp Ridge functions.
    Combines the structural idea of CMA-ES (inspiration 1) with the refinement
    machinery of the original program (finite difference gradient estimation).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(123)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial random sampling to locate the basin
        n_initial = 20
        for _ in range(n_initial):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES with multiple restarts
        def cma_es(start_x, budget_alloc, seed_offset=0):
            nonlocal evaluated, best_x, best_f, deadline, elapsed_ok

            if evaluated >= budget or not elapsed_ok():
                return

            rng_local = np.random.default_rng(42 + seed_offset)

            # CMA-ES parameters
            pop_size = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # ~13 for dim=5
            sigma = 1.0
            mean = start_x.copy().astype(float)
            C = np.eye(dim)

            # Adaptation rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Initial evaluations
            for _ in range(min(pop_size, budget_alloc)):
                if evaluated >= budget or not elapsed_ok():
                    break
                x_candidate = clip(mean + sigma * rng_local.standard_normal(dim))
                f_val = eval_point(x_candidate)

            # Main loop
            while evaluated < budget and elapsed_ok():
                if evaluated + pop_size > budget:
                    break

                # Generate population
                xs = []
                for _ in range(pop_size):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    # Sample from N(mean, sigma^2 * C)
                    z = rng_local.standard_normal(dim)
                    # Cholesky decomposition for sampling
                    try:
                        L = np.linalg.cholesky(C)
                        x_new = clip(mean + sigma * L @ z)
                    except np.linalg.LinAlgError:
                        x_new = clip(mean + sigma * z)
                    f_val = eval_point(x_new)
                    xs.append((x_new, f_val))

                if len(xs) < pop_size:
                    break

                # Sort by fitness
                xs.sort(key=lambda x: x[1])

                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * xs[i][0]

                # Update sigma
                old_mean = mean.copy()
                mean = new_mean

                # Compute direction for sigma adaptation
                try:
                    C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    C_sqrt_inv = np.eye(dim)

                diff = (mean - old_mean) / (sigma * np.sqrt(mu_eff))
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * diff
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)
                if h_sigma < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1) / np.sqrt(dim - 1):
                    sigma *= np.exp(c_sigma / dim * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - np.sqrt(dim - 1)))

                # Update covariance matrix
                diff_c = (mean - old_mean) / sigma
                p_c = (1 - c_c) * p_c + c_c * diff_c

                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2)) / \
                      ((1 + 2 / (dim + 1)) * np.sqrt(dim + 1) / np.sqrt(dim - 1))

                rank_one = np.outer(p_c, p_c)
                if h_c < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1) / np.sqrt(dim - 1):
                    rank_one += (1 - (1 - c_c)**2) * (c_c / (1 - c_c)) * np.eye(dim)

                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (xs[i][0] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    sigma = max(sigma * 0.5, 0.01)

            # After CMA-ES, do local refinement with smaller steps
            if evaluated < budget and elapsed_ok():
                local_sigma = sigma * 0.1
                for _ in range(min(20, budget - evaluated)):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    x_candidate = clip(mean + local_sigma * rng_local.standard_normal(dim))
                    f_val = eval_point(x_candidate)

        # Multi-start strategy
        n_starts = 3
        budget_per_start = budget // n_starts

        # First start: random initialization
        start1 = rng.uniform(-3.0, 3.0, size=dim)
        cma_es(start1, budget_per_start, seed_offset=0)

        # Second start: near the best found so far
        if best_x is not None:
            start2 = best_x + rng.normal(0, 0.5, size=dim)
            cma_es(start2, budget_per_start, seed_offset=100)
        else:
            start2 = rng.uniform(-3.0, 3.0, size=dim)
            cma_es(start2, budget_per_start, seed_offset=100)

        # Third start: use remaining budget
        remaining = budget - evaluated
        if remaining > 0 and elapsed_ok():
            if best_x is not None:
                start3 = best_x + rng.normal(0, 0.3, size=dim)
            else:
                start3 = rng.uniform(-2.0, 2.0, size=dim)
            cma_es(start3, remaining, seed_offset=200)

        # Phase 3: Local refinement using finite difference gradient
        if best_x is not None and evaluated < budget and elapsed_ok():
            x = best_x.copy()
            f = best_f
            h = 0.05  # Smaller step for refinement
            max_iters = 10

            for it in range(max_iters):
                if not elapsed_ok() or evaluated >= budget:
                    break

                # Estimate gradient using central differences
                g = np.zeros(dim)
                for j in range(dim):
                    if not elapsed_ok() or evaluated >= budget:
                        break
                    x_plus = x.copy()
                    x_plus[j] += h
                    x_minus = x.copy()
                    x_minus[j] -= h
                    f_plus = eval_point(x_plus)
                    f_minus = eval_point(x_minus)
                    g[j] = (f_plus - f_minus) / (2 * h)

                if not elapsed_ok() or evaluated >= budget:
                    break

                # Check if gradient is small (convergence)
                if np.linalg.norm(g) < 1e-8:
                    break

                # Line search for step size
                alpha = 1.0
                c1 = 1e-4  # Armijo condition parameter
                rho = 0.5  # Backtracking factor

                for _ in range(10):
                    if not elapsed_ok() or evaluated >= budget:
                        break

                    x_new = clip(x - alpha * g)
                    f_new = eval_point(x_new)

                    if f_new < f - c1 * alpha * np.dot(g, g):
                        # Armijo condition satisfied
                        x = x_new
                        f = f_new
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        break
                    else:
                        alpha *= rho

                # If no step accepted, reduce h and retry
                h *= 0.5
                if h < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
