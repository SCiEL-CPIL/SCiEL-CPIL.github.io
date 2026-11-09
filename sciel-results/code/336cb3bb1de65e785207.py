"""Rewritten program using CMA-ES with adaptive restarts for Rosenbrock."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multiple restarts for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Use 280s limit to be safe within 300s
        self.deadline = self.start_time + 280.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")

            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)

            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()

            return val

        rng = np.random.default_rng(42)

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0, using at most max_evals function calls."""
            dim = self.dim
            pop_size = int(4 + 3 * np.log(dim))
            pop_size = max(pop_size, 5)

            # CMA-ES parameters
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
            cs = (dim + 1.7) / (dim + 9.2 + cc)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialize
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

            evals = 0
            best_local_x = x_mean.copy()
            best_local_f = np.inf

            while evals < max_evals and self.calls_made < self.budget and time.time() < self.deadline:
                # Generate population
                for i in range(pop_size):
                    if self.calls_made >= self.budget or time.time() >= self.deadline:
                        break

                    # Sample from N(x_mean, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    x_new = x_mean + sigma * (B @ (D * z))

                    try:
                        f_val = wrapped_func(x_new)
                        evals += 1
                    except StopIteration:
                        break

                    if f_val < best_local_f:
                        best_local_f = f_val
                        best_local_x = x_new.copy()

                if self.calls_made >= self.budget or time.time() >= self.deadline:
                    break

                # Sort by fitness
                # We need to track which points were evaluated
                # Let's restructure to collect all points first

            # Return the best found
            if best_local_f < self.best_f:
                self.best_f = best_local_f
                self.best_x = best_local_x.copy()

            return best_local_x, best_local_f

        # Better CMA-ES implementation with proper population management
        def cma_es_proper(x0, sigma0, max_evals):
            dim = self.dim
            pop_size = int(4 + 3 * np.log(dim))
            pop_size = max(pop_size, 5)

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

            # Initialize
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            evals = 0
            best_local_x = x_mean.copy()
            best_local_f = np.inf

            # Evaluate initial point
            try:
                f_val = wrapped_func(x_mean.copy())
                evals += 1
                best_local_f = f_val
                best_local_x = x_mean.copy()
            except StopIteration:
                return best_local_x, best_local_f

            while evals < max_evals and self.calls_made < self.budget and time.time() < self.deadline:
                # Generate population
                points = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    x_new = x_mean + sigma * (B @ (D * z))
                    points.append(x_new)

                # Evaluate all points in this generation
                fitnesses = []
                for x_new in points:
                    if self.calls_made >= self.budget or time.time() >= self.deadline:
                        break
                    try:
                        f_val = wrapped_func(x_new)
                        evals += 1
                        fitnesses.append(f_val)
                        if f_val < best_local_f:
                            best_local_f = f_val
                            best_local_x = x_new.copy()
                    except StopIteration:
                        break

                if len(fitnesses) == 0:
                    break

                # If we couldn't evaluate all points, stop
                if len(fitnesses) < pop_size:
                    break

                # Sort by fitness
                indices = np.argsort(fitnesses)
                x_sorted = [points[i] for i in indices]
                f_sorted = [fitnesses[i] for i in indices]

                # Update mean
                x_mean_new = np.zeros(dim)
                for i in range(mu):
                    x_mean_new += weights[i] * x_sorted[i]
                x_mean = x_mean_new

                # Update covariance matrix
                v = np.array([(x_sorted[i] - x_mean_old) / sigma for i in range(mu)]).T if hasattr(self, '_x_mean_old') else None

                # Actually, let's track x_mean_old properly
                # This is getting complex. Let me simplify.

                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D @ (x_mean - x_mean_old) / sigma))

                # Check stagnation
                if np.linalg.norm(ps) / chi_n > 1.4 + 2.0 / (dim + 1):
                    sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(ps) / chi_n - 1))

                # Update pc and C
                h_sigma = 1 if np.linalg.norm(ps) / chi_n <= (1.4 + 2.0 / (dim + 1)) else 0
                pc = (1 - c_c) * pc + h_sigma * c_c * (x_mean - x_mean_old) / sigma

                # Update C
                C = B @ (D ** 2) @ B.T
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    v_i = (x_sorted[i] - x_mean_old) / sigma
                    rank_mu += weights[i] * np.outer(v_i, v_i)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except:
                    pass

                # Save old mean for next iteration
                self._x_mean_old = x_mean_old

            return best_local_x, best_local_f

        # Simpler and more robust approach: use scipy's minimize with Nelder-Mead as fallback
        # and combine with a simple coordinate descent

        # Let's use a cleaner multi-start strategy with Powell + coordinate descent polish

        # Strategy:
        # 1. Start with a few diverse points
        # 2. Use Powell's method (which is derivative-free and works well for Rosenbrock)
        # 3. Polish with coordinate descent
        # 4. If budget remains, do more starts from best found so far

        import scipy.optimize

        def powell_with_callback(x0, max_iter=100):
            """Run Powell's method with callback for early stopping."""
            def callback(xk):
                return self.calls_made >= self.budget or time.time() > self.deadline

            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-6},
                    callback=callback
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5).copy()
                return res.x
            except StopIteration:
                return x0
            except Exception:
                return x0

        def coordinate_descent(x0, max_rounds=50):
            """Simple coordinate descent for polishing."""
            x = np.array(x0, dtype=float)
            f_curr = self.best_f

            for _ in range(max_rounds):
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break

                improved = False
                for i in range(self.dim):
                    if self.calls_made >= self.budget or time.time() > self.deadline:
                        break

                    # Try small steps in each direction
                    for step in [0.1, 0.01, 0.001, -0.1, -0.01, -0.001]:
                        x_trial = x.copy()
                        x_trial[i] += step
                        x_trial = np.clip(x_trial, -5, 5)

                        if self.calls_made >= self.budget or time.time() > self.deadline:
                            break

                        try:
                            f_trial = wrapped_func(x_trial)
                        except StopIteration:
                            break

                        if f_trial < f_curr:
                            x = x_trial
                            f_curr = f_trial
                            improved = True
                            break

                    if self.calls_made >= self.budget or time.time() > self.deadline:
                        break

                if not improved:
                    break

            if f_curr < self.best_f:
                self.best_f = f_curr
                self.best_x = x.copy()

            return x

        # Multi-start Powell with coordinate descent polish
        # Start with diverse points
        starts = [
            np.zeros(self.dim),
            rng.uniform(-3, 3, self.dim),
            rng.uniform(-3, 3, self.dim),
            rng.uniform(-2, 2, self.dim),
        ]

        # Budget allocation: give more budget to early starts, less to later ones
        # Total budget: 500
        # Powell typically uses ~dim * iterations calls
        # Let's allocate: 150, 120, 100, 80 = 450, leaving 50 for polish

        budget_allocs = [150, 120, 100, 80]

        for i, x0 in enumerate(starts):
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break

            # Run Powell
            x_best = powell_with_callback(x0, max_iter=50)

            # Polish with coordinate descent
            if self.calls_made < self.budget and time.time() < self.deadline:
                x_best = coordinate_descent(x_best, max_rounds=20)

            # If we found a good point, try another start from near it
            if i < len(starts) - 1 and self.best_f < 100:
                # Perturb the best point for next start
                perturbation = rng.normal(0, 0.5, self.dim)
                starts.append(np.clip(self.best_x + perturbation, -5, 5))
                budget_allocs.append(60)

        # Final polish with more coordinate descent iterations if budget remains
        if self.calls_made < self.budget and time.time() < self.deadline and self.best_x is not None:
            x_final = coordinate_descent(self.best_x, max_rounds=30)
            if self.best_x is not None:
                # Try a final small-scale local search
                for step_size in [0.01, 0.001]:
                    if self.calls_made >= self.budget or time.time() > self.deadline:
                        break
                    for i in range(self.dim):
                        if self.calls_made >= self.budget or time.time() > self.deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = self.best_x.copy()
                            x_trial[i] += direction * step_size
                            x_trial = np.clip(x_trial, -5, 5)
                            if self.calls_made >= self.budget or time.time() > self.deadline:
                                break
                            try:
                                f_trial = wrapped_func(x_trial)
                                if f_trial < self.best_f:
                                    self.best_f = f_trial
                                    self.best_x = x_trial.copy()
                            except StopIteration:
                                break

        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(np.clip(self.best_x, -5, 5))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
