"""Two-phase Sobol mapping + CMA-ES for Rastrigin-rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Two-phase optimizer for Rastrigin-rotated f15:
    1. Sparse Sobol quasi-random sequence to map landscape and identify top basins
    2. Single sustained CMA-ES instance with large initial step size
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_check():
            return time.time() < deadline and calls_made < budget

        def sobol_sequence(n_samples, dim, seed=42):
            """Generate Sobol quasi-random sequence in [-5, 5]^dim."""
            # Use Halton sequence as a simpler alternative with good distribution
            # Halton is easier to implement and has similar low-discrepancy properties
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            points = []
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(1, n_samples + 1):
                point = []
                for d in range(dim):
                    h = halton(i, primes[d])
                    point.append(-5.0 + 10.0 * h)
                points.append(point)
            return np.array(points)

        def cmaes(x0, max_evals=200):
            """Simple CMA-ES implementation."""
            nonlocal calls_made, best_f, best_x
            n = dim
            lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
            sigma = 2.0  # large initial step size for exploration
            x_mean = x0.copy()
            f_mean = eval_func(x_mean)
            local_best_f = f_mean
            local_best_x = x_mean.copy()

            # Covariance matrix
            C = np.eye(n)
            # Learning rates
            c_sigma = (lambda_ + 2) / (n + 4) ** 2 + 2
            c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
            mu = lambda_ // 2
            w = np.zeros(mu)
            w_sum = np.sum(np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)))
            w[:] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w_sum
            mu_eff = 1 / np.sum(w ** 2)
            c_cov = (2 + 2 * mu_eff / n) / (n + 1.3) ** 2 + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            for gen in range(max_evals // lambda_ + 1):
                if not time_check() or calls_made >= budget:
                    break

                # Generate population
                offspring = []
                offspring_f = []
                for i in range(lambda_):
                    if not time_check() or calls_made >= budget:
                        break
                    # Sample from multivariate normal
                    z = rng.standard_normal(n)
                    # Apply Cholesky decomposition of C
                    L = np.linalg.cholesky(C)
                    y = L @ z
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = eval_func(x_new)
                    offspring.append(x_new)
                    offspring_f.append(f_new)

                if len(offspring) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in idx]
                offspring_f = [offspring_f[i] for i in idx]

                # Update mean
                x_new_mean = np.zeros(n)
                for i in range(mu):
                    x_new_mean += w[i] * offspring[i]
                x_mean = x_new_mean

                # Update step size
                y_mean = (x_mean - x_old_mean) / sigma if hasattr(self, '_cmaes_sigma') else (x_mean - x0) / sigma
                # Simplified: use fixed sigma adaptation
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean / chi_n
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance
                delta_h = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n else 0
                p_c = (1 - c_c) * p_c + delta_h * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                C = (1 - c_cov) * C + c_cov * np.outer(y_mean, y_mean)
                for i in range(mu):
                    C += c_cov * w[i] * np.outer(offspring[i] - x_old_mean, offspring[i] - x_old_mean)

                # Update best
                if offspring_f[0] < local_best_f:
                    local_best_f = offspring_f[0]
                    local_best_x = offspring[0].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

                x_old_mean = x_mean.copy()

            return local_best_x, local_best_f

        # Phase 1: Sobol quasi-random sampling to map landscape
        # Use 60 Sobol points for initial exploration
        n_sobol = 60
        sobol_points = sobol_sequence(n_sobol, dim)

        # Evaluate all Sobol points
        sobol_results = []
        for sp in sobol_points:
            if not time_check() or calls_made >= budget:
                break
            fx = eval_func(sp)
            sobol_results.append((sp, fx))

        # Sort by fitness and get top 3
        sobol_results.sort(key=lambda x: x[1])
        top_basins = [x[0] for x in sobol_results[:3]]

        # Phase 2: CMA-ES from the best identified basin with large initial step size
        if time_check() and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 30:
                cmaes_evals = min(remaining, 300)
                bx, bf = cmaes(best_x.copy(), max_evals=cmaes_evals)
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        # Phase 3: If budget remains, try CMA-ES from second and third best basins
        if time_check() and len(top_basins) > 1 and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 30:
                cmaes_evals = min(remaining // 2, 100)
                for basin in top_basins[1:3]:
                    if not time_check() or calls_made >= budget:
                        break
                    if remaining < 30:
                        break
                    bx, bf = cmaes(basin.copy(), max_evals=cmaes_evals)
                    if bf < best_f:
                        best_f = bf
                        best_x = bx.copy()
                    remaining = budget - calls_made

        # Phase 4: Final fine-tuning with small-step CMA-ES if budget remains
        if time_check() and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 20:
                # Use Nelder-Mead for final fine-tuning
                try:
                    result = minimize(
                        lambda x: eval_func(x),
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(50, remaining // 5),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
