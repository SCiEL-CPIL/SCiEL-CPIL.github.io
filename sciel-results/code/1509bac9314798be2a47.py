"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_calls):
            """CMA-ES with adaptive covariance matrix."""
            nonlocal total_calls, best_x, best_f
            if max_calls < 10:
                return x0, best_f

            # Parameters
            mu = max(1, (n + 1) // 2)
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            w = w**2
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

            # State
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Initialize population
            pop_size = max(4, 4 + int(3 * np.log(n)))
            pop_size = min(pop_size, max_calls // 2)

            for gen in range(max_calls):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                # Generate population
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    # Use Cholesky decomposition for C^{1/2}
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    x_trial = x + sigma * (L @ z)
                    f_trial = try_eval(x_trial)
                    if f_trial is None:
                        break
                    # Store for ranking
                    if i == 0:
                        pop = [(x_trial, f_trial)]
                    else:
                        pop.append((x_trial, f_trial))
                else:
                    # All evaluations succeeded
                    pass

                if total_calls >= budget or time.time() > deadline - 5:
                    break

                # Rank by function value
                pop.sort(key=lambda t: t[1])
                selected = pop[:min(mu, len(pop))]
                
                if len(selected) < 1:
                    break

                # Update mean
                x_new = np.sum([w[i] * selected[i][0] for i in range(len(selected))], axis=0)

                # Update sigma
                x_old = x
                x = x_new
                if np.linalg.norm(x - x_old) > 1e-14:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (x - x_old) / sigma
                    if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) > (1.4 + 2 / (n + 1)) * chi_n:
                        sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance
                diff = [(x - x_old) / sigma for _ in range(len(selected))]
                diff = [selected[i][0] - x_old for i in range(len(selected))]
                diff = [d / sigma for d in diff]
                
                # Outer product sum
                outer_sum = np.zeros((n, n))
                for i in range(len(selected)):
                    outer_sum += w[i] * np.outer(diff[i], diff[i])

                h_sigma = 1.0
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) > (1.4 + 2 / (n + 1)) * chi_n:
                    h_sigma = 0.0

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x - x_old) / sigma

                C = (1 + c_1 + c_mu - c_1 - c_1 / n) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * outer_sum

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)

                # Check convergence
                if np.linalg.norm(x - x_old) < 1e-10 * sigma:
                    break

            return x, best_f

        # --- Phase 1: Initial sampling ---
        n_initial = min(30, budget // 8)
        structured_points = [
            np.zeros(n),
            np.ones(n) * 2,
            -np.ones(n) * 2,
            np.ones(n) * 1,
            -np.ones(n) * 1,
            np.array([3.0, -3.0, 2.0, -2.0, 1.0]),
            np.array([-2.0, 3.0, -1.0, 2.0, -3.0]),
            np.array([1.5, 1.5, 1.5, 1.5, 1.5]),
            np.array([-1.5, -1.5, -1.5, -1.5, -1.5]),
        ]
        for p in structured_points:
            if total_calls >= budget:
                break
            try_eval(p)

        for i in range(n_initial):
            if total_calls >= budget or time.time() > deadline - 5:
                break
            scale = 3.5 * (1.0 - i / max(1, n_initial)) + 0.5
            x_try = rng.normal(0, scale, n)
            try_eval(x_try)

        if best_x is None:
            return None, np.inf

        # --- Phase 2: Multi-start CMA-ES ---
        # Start from best point and a few diverse points
        starts = [best_x.copy()]
        # Add diverse starts
        for _ in range(4):
            x_rand = rng.uniform(-4, 4, n)
            f_rand = try_eval(x_rand)
            if f_rand is not None:
                starts.append(x_rand.copy())

        # Run CMA-ES on each start
        remaining_budget = budget - total_calls
        budget_per_start = max(20, remaining_budget // len(starts))
        for i, x_start in enumerate(starts):
            if total_calls >= budget - 10 or time.time() > deadline - 10:
                break
            remaining = budget - total_calls
            alloc = min(budget_per_start, remaining)
            if alloc < 15:
                break
            sigma0 = 1.0 if i == 0 else 0.5
            x_opt, f_opt = cma_es(x_start, sigma0, alloc)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # --- Phase 3: Final polish ---
        if total_calls < budget and best_x is not None and time.time() < deadline - 5:
            # Small CMA-ES from best point
            if budget - total_calls >= 30:
                x_opt, f_opt = cma_es(best_x.copy(), 0.1, min(50, budget - total_calls))
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

            # Tiny perturbations
            n_final = min(10, budget - total_calls)
            for i in range(n_final):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                perturbation = rng.normal(0, 0.01, n)
                x_try = best_x + perturbation
                f_try = try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
