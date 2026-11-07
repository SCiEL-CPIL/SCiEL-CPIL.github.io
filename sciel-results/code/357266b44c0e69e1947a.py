"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _sobol_sequence(n, dim, seed=42):
    """Generate a quasi-random Sobol-like sequence using Halton as a robust fallback."""
    # Use a simple Halton sequence as a deterministic quasi-random sequence
    points = np.zeros((n, dim))
    for d in range(dim):
        base = d + 2  # primes: 2, 3, 5, ...
        for i in range(n):
            f = 1.0
            i_val = i + 1
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                points[i, d] += rem * f
    return points


class Optimizer:
    """Two-phase optimizer: Sobol global exploration + CMA-ES-like local exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Sparse Sobol/Halton global exploration
        n_global = min(80, budget // 4)
        halton_points = _sobol_sequence(n_global, dim, seed=42)
        # Scale from [0,1]^dim to [-5,5]^dim
        halton_points = halton_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(n_global):
            if time.time() > deadline:
                break
            x = np.clip(halton_points[i], -5, 5)
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness and take top 3 basins
        candidates.sort(key=lambda t: t[0])
        top_basins = [x.copy() for _, x in candidates[:min(3, len(candidates))]]

        # Phase 2: CMA-ES-like local exploitation from the best basin
        # Use a population-based evolutionary strategy with covariance adaptation
        remaining_budget = budget - n_global - 1  # -1 for initial best evaluation
        if remaining_budget < 10:
            remaining_budget = max(10, budget - n_global)

        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        f_mean = best_f
        sigma = 1.5  # Large initial step size to exploit the basin
        pop_size = 4 + int(3 * np.log(dim))  # ~15 for dim=5

        # CMA-ES state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = (pop_size + 2) / (dim + 4)
        c_c = (4 + dim / 4) / (dim + 4 + pop_size)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_1 = (2 / ((dim + 1.3) ** 2 + pop_size)) / mu_eff
        c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + pop_size)) / mu_eff

        calls_used = n_global
        best_x_es = best_x.copy()
        best_f_es = best_f

        for gen in range(remaining_budget // pop_size + 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Sample population
            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline:
                    break

                # Generate candidate using CMA-ES
                z = rng.standard_normal(dim)
                # Cholesky decomposition for sampling from covariance
                try:
                    L = np.linalg.cholesky(C)
                    x_child = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    x_child = x_mean + sigma * z

                x_child = np.clip(x_child, -5, 5)
                f_child = func(x_child)
                calls_used += 1

                if f_child < best_f_es:
                    best_f_es = f_child
                    best_x_es = x_child.copy()

                # Store for update
                if not hasattr(self, '_cma_buffer'):
                    self._cma_buffer = []
                self._cma_buffer.append((f_child, x_child))

            if len(self._cma_buffer) < pop_size:
                self._cma_buffer = []
                continue

            # Sort by fitness
            self._cma_buffer.sort(key=lambda t: t[0])
            selected = self._cma_buffer[:mu]
            self._cma_buffer = []

            # Update mean
            new_mean = np.sum(np.array([w * x for w, x in zip(weights, selected)]), axis=0)

            # Update sigma (1/5 rule adaptation)
            d_sigma = np.sqrt((1 + 2 / (dim + 1)) * c_sigma * (1 - c_sigma))
            for _, x in selected:
                diff = (x - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * diff / d_sigma

            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used - n_global + 1))) < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1) - 1 else 0.0

            # Update C
            p_c = (1 - c_c) * p_c + c_c * h_sigma * (new_mean - x_mean) / sigma
            delta_h = (1.0 - (1 - c_c) ** (2 * (calls_used - n_global + 1))) if h_sigma else 0.0
            C = ((1 + c_1 * (2 - c_1) - c_mu) * C
                 + c_1 * np.outer(p_c, p_c) / (1 - (1 - c_c) ** (2 * (calls_used - n_global + 1)))
                 + c_mu * np.sum([w * np.outer((x - x_mean) / sigma, (x - x_mean) / sigma)
                                   for w, x in zip(weights, selected)], axis=0))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim + 1) - 1))
            sigma = max(1e-8, min(sigma, 10.0))

            x_mean = new_mean
            f_mean = func(x_mean) if calls_used < budget else f_mean
            if calls_used < budget:
                calls_used += 1
                if f_mean < best_f_es:
                    best_f_es = f_mean
                    best_x_es = x_mean.copy()

        # Final polish with (1+1)-ES if budget remains
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline:
            sigma_polish = max(0.1, sigma * 0.3)
            x_cur = best_x_es.copy()
            f_cur = best_f_es
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x_cur + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < f_cur:
                    x_cur, f_cur = y, fy
                    sigma_polish *= 1.2
                else:
                    sigma_polish *= 0.9
            if f_cur < best_f_es:
                best_x_es, best_f_es = x_cur, f_cur

        return best_x_es, best_f_es
# EVOLVE-BLOCK-END
