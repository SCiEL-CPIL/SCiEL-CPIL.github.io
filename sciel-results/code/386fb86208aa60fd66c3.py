"""CMA-ES Optimizer with stochastic tunneling for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with stochastic tunneling for robust multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _eval(self, func, x):
        """Evaluate and track best."""
        x = np.clip(x, -5.0, 5.0)
        fx = float(func(x))
        self.calls += 1
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def _cma_es(self, func, x0, sigma, max_evals, deadline):
        """Run CMA-ES from starting point x0 with initial step size sigma."""
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))

        # CMA-ES parameters
        cc = (4 + n / (n + 4)) / ((n + 1.3) + (4 + n / (n + 4)))
        cs = (n + 1.5) / (n + 3 + 2 * n / (n + 4))
        cmu = min(1 - cc, 2 * (n - 2 + 1. / n) / ((n + 2.) ** 2 + 2))
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

        # State
        m = x0.copy()
        C = np.eye(n)
        sigma = sigma
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Precompute Cholesky of C
        C_chol = np.linalg.cholesky(C)

        # Track convergence
        no_improve = 0
        best_f = self.best_f

        while self.calls < max_evals and time.time() < deadline:
            # Generate population
            z = np.random.standard_normal((pop_size, n))
            # Transform to y
            y = np.array([z[i] @ C_chol for i in range(pop_size)])
            # Generate candidates
            x_pop = m + sigma * y
            x_pop = np.clip(x_pop, -5.0, 5.0)

            # Evaluate population
            f_pop = np.array([self._eval(func, x_pop[i]) for i in range(pop_size)])

            # Sort by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                no_improve = 0
            else:
                no_improve += 1

            # Check for stagnation - restart if no improvement for too long
            if no_improve > 10:
                break

            # Update mean
            weights = np.full(pop_size, 1.0 / (2 * pop_size))
            weights[0] = 1.0 / 2.0  # Give more weight to best
            mu_eff = 1.0 / np.sum(weights ** 2)
            m_new = np.sum(weights[:, np.newaxis] * x_sorted, axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (m_new - m)
            if np.linalg.norm(ps) / chi_n < (1.4 + cc) * np.sqrt((1 - cs) / (1 + cs)):
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (m_new - m) / sigma

            # Update covariance matrix
            delta_h = 1.0 if np.linalg.norm(ps) / chi_n <= (1.4 + cc) * np.sqrt(
                (1 - cs) / (1 + cs)
            ) else 0.0
            C = (1 + cs * (cc + 2 * cmu) - cs - cc) * C
            C += cs * (ps @ ps + (cc * (1 - cc) - cmu) * (pc @ pc))
            C += cmu / sigma ** 2 * np.sum(
                weights[:, np.newaxis] * np.outer(x_sorted - m, x_sorted - m), axis=0
            )

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                # Reset if Cholesky fails
                C = np.eye(n)
                C_chol = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = max(sigma * 2, 0.1)

            m = m_new

            # Adapt sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Check convergence
            if sigma < 1e-6:
                break

        return m, best_f

    def _stochastic_tunneling(self, func, x0, max_evals, deadline):
        """Stochastic tunneling: perturb best, on rejection shift toward origin."""
        n = self.dim
        x = x0.copy()
        f_x = self.best_f
        step_size = 0.5
        no_improve = 0

        while self.calls < max_evals and time.time() < deadline:
            # Generate perturbation
            noise = np.random.standard_normal(n) * step_size
            x_trial = x + noise
            x_trial = np.clip(x_trial, -5.0, 5.0)
            f_trial = self._eval(func, x_trial)

            if f_trial < f_x:
                # Accept: move to better point
                x = x_trial.copy()
                f_x = f_trial
                step_size = min(step_size * 1.05, 2.0)
                no_improve = 0
            else:
                # Reject: apply deterministic shift toward origin
                # Scale shift by perturbation magnitude to exploit bi-modal structure
                shift_mag = np.linalg.norm(noise)
                shift = -0.3 * shift_mag * x / (np.linalg.norm(x) + 1e-8)
                x = x + shift
                x = np.clip(x, -5.0, 5.0)
                step_size = max(step_size * 0.95, 0.05)
                no_improve += 1

            if no_improve > 30:
                break

        return x, f_x

    def __call__(self, func):
        rng = np.random.default_rng(42)

        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        # Multi-start with hybrid CMA-ES + stochastic tunneling
        # The Lunacek function is bimodal, so we need to explore different regions

        # Starting points: origin and several random points
        start_points = [np.zeros(self.dim)]

        # Add random starting points to explore different basins
        for _ in range(3):
            start_points.append(rng.uniform(-3.0, 3.0, self.dim))

        # Allocate budget: give most to the first (origin) start, rest to others
        budget_per_start = [self.budget // 3, (self.budget - self.budget // 3) // 3] * 3
        # Adjust to ensure we don't exceed budget
        total_allocated = sum(budget_per_start)
        if total_allocated > self.budget:
            scale = self.budget / total_allocated
            budget_per_start = [int(b * scale) for b in budget_per_start]

        for i, x0 in enumerate(start_points):
            if self.calls >= self.budget or time.time() >= deadline:
                break

            remaining_budget = self.budget - self.calls
            alloc = min(budget_per_start[i], remaining_budget)

            if alloc < 10:
                break

            # Phase 1: CMA-ES for structured exploration
            cma_budget = int(alloc * 0.7)
            if cma_budget >= 10:
                sigma = 1.5 if i == 0 else 2.0
                m, _ = self._cma_es(func, x0, sigma, self.calls + cma_budget, deadline)

                # Update x0 to the CMA-ES result for tunneling
                if self.best_x is not None:
                    x0 = self.best_x.copy()

            # Phase 2: Stochastic tunneling for local refinement and escape
            tunnel_budget = alloc - cma_budget
            if tunnel_budget >= 5:
                self._stochastic_tunneling(func, x0, self.calls + tunnel_budget, deadline)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
