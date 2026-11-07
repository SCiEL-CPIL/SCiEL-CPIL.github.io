"""Combined optimizer for BBOB f21 (Gallagher 101 peaks): CMA-ES + adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance adaptation + adaptive local refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 330s kill

        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_func(x):
            nonlocal best_f, best_x, call_count
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64).clip(-5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return best_f if best_f != float('inf') else 1e10
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Coarse random sampling to find good starting region
        # f21 has 101 peaks, need decent exploration but CMA-ES handles the rest
        n_random = min(40, max(15, budget // 12))
        for i in range(n_random):
            if call_count >= budget or time.time() > deadline - 60:
                break
            x = rng.uniform(-4, 4, dim)
            safe_func(x)

        # Try a few structured points
        special_starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
        ]
        for s in special_starts:
            if call_count >= budget or time.time() > deadline - 60:
                break
            safe_func(s)

        if best_x is None:
            best_x = np.zeros(dim)
            safe_func(best_x)

        # Phase 2: CMA-ES with proper population-based covariance adaptation
        # This handles the hidden rotation in f21's peaks
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda = 4 + 3*ln(dim)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Reserve budget for Phase 3 refinement
        phase3_reserve = min(80, max(30, budget // 8))
        phase2_budget = budget - call_count - phase3_reserve
        if phase2_budget < 20:
            phase2_budget = max(0, budget - call_count)

        no_improve_steps = 0
        max_no_improve = 30  # break out of CMA-ES if no improvement for this many steps

        for step in range(phase2_budget):
            if time.time() > deadline - 30:
                break
            if call_count >= budget:
                break

            # Generate population
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 0))  # ensure non-negative
            population = np.empty((pop_size, dim))
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                population[i] = x_mean + sigma * (B * D) @ z

            # Evaluate population
            fitness = np.empty(pop_size)
            for i in range(pop_size):
                if call_count >= budget or time.time() > deadline - 30:
                    fitness[i:] = best_f
                    break
                x_eval = np.clip(population[i], -5, 5)
                fitness[i] = safe_func(x_eval)

            # Sort by fitness
            order = np.argsort(fitness)
            fitness_sorted = fitness[order]
            x_selected = population[order[:mu]]

            # Update best
            if fitness_sorted[0] < best_f:
                best_f = fitness_sorted[0]
                best_x = np.clip(x_selected[0], -5, 5).copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # Check for stagnation
            if no_improve_steps >= max_no_improve:
                break

            # Update mean
            x_new_mean = (weights @ x_selected.T).T

            # Update evolution paths
            y = (x_new_mean - x_mean) / max(sigma, 1e-10)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            # h_sigma: check if p_sigma is within expected range
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_n) < (1.4 + 2 / (dim + 1)) else 0.0

            # Update covariance matrix
            delta_h = h_sigma * c_sigma * (2 - c_sigma)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * sum(w * np.outer(x_selected[i] - x_mean, x_selected[i] - x_mean) for i, w in enumerate(weights))

            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-10:
                C += 1e-8 * np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            x_mean = x_new_mean

        # Phase 3: Adaptive local refinement around the best point
        # Squeeze out final precision with adaptive sigma local search
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            sigma_ref = 0.5
            max_ref_steps = budget - call_count
            ref_no_improve = 0
            max_ref_no_improve = 50

            for step in range(max_ref_steps):
                if time.time() > deadline - 5:
                    break
                if call_count >= budget:
                    break

                y = np.clip(best_x + sigma_ref * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)

                if fy < best_f:
                    sigma_ref *= 1.3
                    ref_no_improve = 0
                else:
                    sigma_ref *= 0.7
                    ref_no_improve += 1

                if sigma_ref < 0.001 or ref_no_improve >= max_ref_no_improve:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
