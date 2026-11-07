"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES global search + L-BFGS-B polish for Rosenbrock (BBOB f8)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        time_limit = 280.0
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise StopIteration
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() - start_time < time_limit

        # --- Phase 1: CMA-ES for global search ---
        # CMA-ES adapts its search distribution to the Rosenbrock valley curvature
        # via covariance matrix updates, enabling faster basin convergence.
        pop_size = 10 + int(3 * dim)  # 25 for dim=5
        sigma = 2.0  # initial step size
        max_cma_iters = 40  # limit CMA-ES iterations to leave budget for polish

        # Initial mean: center of search space
        mean = np.zeros(dim)
        # Initial covariance: identity
        C = np.eye(dim)
        # Inverse Cholesky for sampling
        L_inv = np.linalg.cholesky(C)

        # Evaluate initial mean
        try:
            safe_call(mean)
        except StopIteration:
            pass

        cma_iter = 0
        while cma_iter < max_cma_iters:
            if not check_time() or calls_made >= budget - 30:
                break

            # Generate population
            population = []
            for i in range(pop_size):
                if not check_time() or calls_made >= budget - 30:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (L_inv @ z)
                x = np.clip(x, -5, 5)
                population.append(x)

            # Evaluate population
            pop_f = []
            for x in population:
                if not check_time() or calls_made >= budget - 30:
                    break
                try:
                    pop_f.append(safe_call(x))
                except StopIteration:
                    break

            if len(pop_f) < pop_size // 2:
                break

            # Select top lambda/2
            lam = len(pop_f)
            mu = lam // 2
            indices = np.argsort(pop_f)[:mu]
            selected_x = [population[i] for i in indices]
            selected_f = [pop_f[i] for i in indices]

            # Weighted recombination
            weights = np.exp(-np.arange(1, mu + 1) / (mu / 2.0))
            weights /= weights.sum()

            new_mean = np.average(selected_x, axis=0, weights=weights)
            new_mean = np.clip(new_mean, -5, 5)

            # Update covariance matrix (simplified: rank-1 update)
            diff = new_mean - mean
            C = C + (1 / (mu * mu)) * np.outer(diff, diff)
            # Ensure positive definiteness
            C = (C + C.T) / 2 + 1e-8 * np.eye(dim)

            # Update sigma (simplified adaptation)
            sigma *= 0.95

            # Update inverse Cholesky
            try:
                L_inv = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L_inv = np.linalg.cholesky(C + 1e-6 * np.eye(dim))

            mean = new_mean
            cma_iter += 1

        # --- Phase 2: L-BFGS-B polish from the best point ---
        if check_time() and calls_made < budget - 5 and best_x is not None:
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= budget:
                    raise StopIteration
                calls_made += 1
                return float(func(x_in))

            try:
                from scipy.optimize import minimize
                res = minimize(lbfun, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(dim)],
                               options={'maxiter': 300, 'maxfun': budget - calls_made,
                                        'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # --- Phase 3: Second L-BFGS-B from perturbed best if still not optimal ---
        if check_time() and calls_made < budget - 5 and best_f > 0.1 and best_x is not None:
            x2 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
            try:
                safe_call(x2)

                def lbfun2(x_in):
                    nonlocal calls_made
                    if calls_made >= budget:
                        raise StopIteration
                    calls_made += 1
                    return float(func(x_in))

                from scipy.optimize import minimize
                res2 = minimize(lbfun2, x2, method='L-BFGS-B',
                                bounds=[(-5.0, 5.0) for _ in range(dim)],
                                options={'maxiter': 200, 'maxfun': budget - calls_made,
                                         'gtol': 1e-8})
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = res2.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # --- Phase 4: Final short random refinement ---
        if check_time() and calls_made < budget - 2 and best_x is not None:
            for _ in range(min(30, budget - calls_made)):
                if not check_time() or calls_made >= budget:
                    break
                x3 = np.clip(best_x + 0.1 * rng.standard_normal(dim), -5, 5)
                try:
                    safe_call(x3)
                except StopIteration:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
