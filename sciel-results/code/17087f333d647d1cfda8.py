import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 285  # 15s margin before 300s kill
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if self.calls >= budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            self.calls += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if np.isfinite(f) and f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # CMA-ES implementation for 5D
        # Parameters tuned for dim=5, budget=500

        def cma_es(start_x, budget_alloc):
            """Run CMA-ES from start_x with given function call budget."""
            n = dim
            # Initial parameters
            sigma = 1.0
            popsize = 4 + int(3 * np.log(n))  # lambda = 10 for n=5
            popsize = min(popsize, 12)

            # CMA-ES parameters
            mu = popsize // 2  # number of recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 - mueff * 0.1) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            mean = np.array(start_x, dtype=float)
            C = np.eye(n)
            C_sqrt = np.eye(n)
            ps = np.zeros(n)
            ph = np.zeros(n)

            best_x_local = mean.copy()
            best_f_local = np.inf

            calls_used = 0

            # Cholesky helper
            def cholesky(mat):
                try:
                    return np.linalg.cholesky(mat)
                except np.linalg.LinAlgError:
                    # Regularize
                    return np.linalg.cholesky(mat + 1e-10 * np.eye(n))

            # Generate population
            while calls_used < budget_alloc and time.time() < self.deadline - 2:
                # Sample population
                offspring = []
                for i in range(popsize):
                    z = np.random.randn(n)
                    x = mean + sigma * (C_sqrt @ z)
                    offspring.append(x)

                # Evaluate
                offspring_f = []
                for x in offspring:
                    f = wrapped_func(x)
                    calls_used += 1
                    offspring_f.append(f)
                    if f < best_f_local:
                        best_f_local = f
                        best_x_local = x.copy()

                if not np.isfinite(best_f_local):
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring_sorted = [offspring[i] for i in idx]
                offspring_f_sorted = [offspring_f[i] for i in idx]

                # Update mean (weighted recombination)
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * offspring_sorted[i]
                mean = new_mean

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((C_sqrt.T @ (offspring_sorted[0] - mean)) / sigma))
                # Simplified: use old C for the update
                # Actually need to be more careful. Let's use a simplified CMA-ES.

                # Update covariance matrix
                # This is getting complex. Let's use a simpler variant.
                
                if calls_used >= budget_alloc - 5:
                    break

            return best_x_local, best_f_local

        # Simpler and more robust approach: Use scipy's L-BFGS-B with multiple starts
        # combined with a simple population-based search

        # Phase 1: Generate diverse starting points
        np.random.seed(42)
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(8):
            starts.append(np.random.uniform(-4, 4, dim))
        for _ in range(4):
            starts.append(np.random.uniform(-5, -3, dim))
            starts.append(np.random.uniform(3, 5, dim))

        # Phase 2: For each start, run a local optimizer with limited budget
        # Use L-BFGS-B which is efficient for smooth functions
        calls_per_opt = max(30, (budget - 20) // len(starts))

        for i, x0 in enumerate(starts):
            if self.calls >= budget - 10:
                break
            if time.time() > self.deadline - 10:
                break

            x0 = np.clip(x0, -5, 5)

            # First evaluate the starting point
            f0 = wrapped_func(x0)
            if not np.isfinite(f0):
                continue

            # Run L-BFGS-B with limited iterations
            try:
                maxiter = min(calls_per_opt // 5, 50)
                res = minimize(wrapped_func, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12})
            except Exception:
                res = None

            if res is not None and np.isfinite(res.fun):
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5).copy()

        # Phase 3: Polish the best point with Nelder-Mead
        if self.calls < budget - 5 and time.time() < self.deadline - 5:
            try:
                remaining = budget - self.calls - 2
                maxiter = max(1, min(remaining // 10, 100))
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-14,
                                        'adaptive': True})
                if np.isfinite(res.fun) and res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 4: If budget remains, do a few more random restarts around best
        if self.calls < budget - 10 and time.time() < self.deadline - 10:
            for _ in range(5):
                if self.calls >= budget - 5 or time.time() > self.deadline - 5:
                    break
                # Perturb best point
                noise = np.random.randn(dim) * 0.1
                x_pert = np.clip(self.best_x + noise, -5, 5)
                f_pert = wrapped_func(x_pert)
                if np.isfinite(f_pert) and f_pert < self.best_f:
                    self.best_f = f_pert
                    self.best_x = x_pert.copy()

                # Quick local polish
                try:
                    maxiter = max(1, min((budget - self.calls) // 20, 20))
                    if maxiter > 1:
                        res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-15})
                        if np.isfinite(res.fun) and res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        return self.best_x, self.best_f
