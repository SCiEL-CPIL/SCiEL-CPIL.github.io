"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class CMAES:
    """CMA-ES implementation for continuous optimization."""

    def __init__(self, x0, sigma0, dim, rng):
        self.dim = dim
        self.x = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.rng = rng

        # Population parameters
        self.lam = max(4, int(4 + 3 * dim))  # population size
        self.mu = self.lam // 2  # number of parents

        # Weights for recombination
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= weights.sum()
        self.weights = weights
        self.mu_eff = 1.0 / (weights ** 2).sum()

        # Step size control
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma

        # Covariance matrix adaptation
        self.c1 = (2 - self._cmua() + self.mu_eff / (dim + 1.3)) / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, (self.mu_eff - 2 + 2 / (self.mu_eff + 1)) /
                       ((dim + 2) ** 2 + self.mu_eff))
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.dc = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1) - 1)) + self.cc

        # State
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.chisq = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        self.count_evals = 0
        self.evals_per_iter = 0

    def _cmua(self):
        return (self.mu_eff + 2) / (self.dim ** 2 + self.mu_eff)

    def step(self, func, deadline, budget_remaining):
        """Perform one CMA-ES iteration. Returns (x, fx, evals_used, improved)."""
        if self.evals_per_iter == 0:
            self.evals_per_iter = self.lam

        if self.evals_per_iter > budget_remaining:
            self.evals_per_iter = max(1, budget_remaining)

        # Generate population
        C_sqrt = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.dim))
        samples = self.rng.standard_normal((self.evals_per_iter, self.dim))
        pop = self.x + self.sigma * (samples @ C_sqrt.T)

        # Evaluate population
        fvals = np.empty(self.evals_per_iter)
        for i in range(self.evals_per_iter):
            pop[i] = np.clip(pop[i], -5, 5)
            fvals[i] = func(pop[i])

        self.count_evals += self.evals_per_iter

        # Sort by fitness
        order = np.argsort(fvals)
        pop_sorted = pop[order]
        fvals_sorted = fvals[order]

        # Mean update
        w = self.weights[:min(self.mu, self.evals_per_iter)]
        w = w / w.sum()
        mean_old = self.x.copy()
        self.x = (w[:, None] * pop_sorted[:len(w)]).sum(axis=0)

        # Step size control
        self.ps = (1 - self.c_sigma) * self.ps + \
                  np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * \
                  (C_sqrt.T @ ((self.x - mean_old) / (self.sigma * C_sqrt.diagonal())))
        if np.linalg.norm(self.ps) / self.chisq < (1.4 + 2 / (self.dim + 1)) * np.sqrt((1 - (1 - self.c_sigma)) ** self.count_evals / self.evals_per_iter + 1):
            pass
        # Simplified: just update sigma
        self.sigma *= np.exp(self.c_sigma / self.d_sigma *
                             (np.linalg.norm(self.ps) / self.chisq - 1))
        self.sigma = np.clip(self.sigma, 1e-8, 1e6)

        # Covariance matrix update
        y = (self.x - mean_old) / self.sigma
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y
        delta_sig = (np.linalg.norm(self.ps) / self.chisq - 1) * (1 + self.d_sigma)
        self.C = (1 + self.c1 * delta_sig + self.cmu - self.c1) * self.C + \
                 self.c1 * (self.pc @ self.pc.T - self.C) + \
                 self.cmu * sum(w[i] * np.outer(pop_sorted[i] - mean_old, pop_sorted[i] - mean_old)
                                 for i in range(len(w))) / self.sigma ** 2

        # Ensure C is symmetric and positive definite
        self.C = (self.C + self.C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-10, None)
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Update evals counter
        self.evals_per_iter = self.lam
        self.count_evals += 0  # Already counted above

        best_fx = fvals_sorted[0]
        best_x = pop_sorted[0].copy()

        return best_x, best_fx, self.evals_per_iter


class Optimizer:
    """CMA-ES with multiple restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf
        calls_used = 0

        # Phase 1: Initial exploration with a few random starts to find a good basin
        n_starts = min(5, self.budget // 50)
        for i in range(n_starts):
            if calls_used >= self.budget or time.time() > deadline:
                break
            x0 = rng.uniform(-4, 4, self.dim)
            fx0 = func(x0)
            calls_used += 1
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()

        # Phase 2: CMA-ES from the best starting point
        if best_x is not None and calls_used < self.budget and time.time() < deadline:
            sigma0 = 2.0
            cma = CMAES(best_x, sigma0, self.dim, rng)

            stagnation_counter = 0
            stagnation_threshold = 20  # iterations without improvement

            while calls_used < self.budget and time.time() < deadline:
                budget_remaining = self.budget - calls_used
                if budget_remaining < cma.lam:
                    break

                prev_best = best_fx
                x_new, fx_new, evals_used = cma.step(func, deadline, budget_remaining)
                calls_used += evals_used

                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # If CMA-ES stagnates, try a restart from a different point
                if stagnation_counter >= stagnation_threshold:
                    # Try a local restart around the best point
                    if calls_used + 2 < self.budget and time.time() < deadline:
                        x_restart = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                        fx_restart = func(x_restart)
                        calls_used += 1
                        if fx_restart < best_fx:
                            best_fx = fx_restart
                            best_x = x_restart.copy()

                    # Reset CMA-ES with smaller sigma
                    if calls_used < self.budget and time.time() < deadline:
                        cma = CMAES(best_x, best_fx * 0.1 + 0.5, self.dim, rng)
                        stagnation_counter = 0

        # Phase 3: Fine-tuning with small steps if budget remains
        if calls_used < self.budget and best_x is not None and time.time() < deadline:
            small_sigma = 0.05
            for _ in range(min(50, self.budget - calls_used)):
                if calls_used >= self.budget or time.time() > deadline:
                    break
                y = np.clip(best_x + small_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    small_sigma *= 0.9
                else:
                    small_sigma *= 0.8

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_used += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
