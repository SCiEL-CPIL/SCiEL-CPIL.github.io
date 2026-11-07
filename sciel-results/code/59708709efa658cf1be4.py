"""Rewritten optimizer using Multi-Start Nelder-Mead with CMA-ES initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with CMA-ES initialization for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_f = np.inf
        num_calls = 0

        def check_time():
            return time.time() > deadline - 10

        def evaluate(x):
            nonlocal num_calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            num_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, max_iter=200):
            """Nelder-Mead simplex method."""
            nonlocal num_calls, best_x, best_f
            if check_time() or num_calls >= self.budget:
                return

            d = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = evaluate(x0)
            for i in range(d):
                xi = x0.copy()
                if abs(xi[i]) < 1e-8:
                    xi[i] += 0.25
                else:
                    xi[i] += 0.25 * np.sign(xi[i])
                simplex.append(xi)

            # Evaluate simplex
            fs = [f0]
            for i in range(1, d + 1):
                if check_time() or num_calls >= self.budget:
                    return
                fs.append(evaluate(simplex[i]))

            # Nelder-Mead iterations
            for it in range(max_iter):
                if check_time() or num_calls >= self.budget:
                    return

                # Sort by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)

                if fr < fs[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                    fc = evaluate(xc)
                    if fc < min(fr, fs[-1]):
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if check_time() or num_calls >= self.budget:
                                return
                            fs[i] = evaluate(simplex[i])

                # Check convergence
                if np.std(fs) < 1e-10:
                    break

        def cma_es_init(x0, n_iter=50):
            """CMA-ES for initialization phase."""
            nonlocal num_calls, best_x, best_f
            if check_time() or num_calls >= self.budget:
                return

            d = self.dim
            mean = x0.copy()
            sigma = 1.0

            lambda_ = 4 + int(3 * np.log(d))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            c1 = (2 * (mueff + 0.5)) / ((d + 2)**2 + mueff + 5)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff + 5))
            cs = (mueff + 1.7) / (d + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
            c_sigma = cs * (ds + 1)
            chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))

            C = np.eye(d)
            eigenvalues = np.ones(d)
            eigenvectors = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)

            for it in range(n_iter):
                if check_time() or num_calls >= self.budget:
                    return

                # Generate offspring
                offspring = []
                fitness = []
                for i in range(lambda_):
                    if num_calls >= self.budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    x_i = mean + sigma * (eigenvectors @ (np.sqrt(eigenvalues) * z))
                    f_i = evaluate(x_i)
                    offspring.append(x_i)
                    fitness.append(f_i)

                if len(offspring) < mu:
                    break

                # Sort
                sorted_indices = np.argsort(fitness)
                offspring = [offspring[i] for i in sorted_indices]
                fitness = [fitness[i] for i in sorted_indices]

                old_mean = mean.copy()

                # Update mean
                new_mean = np.zeros(d)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i]

                # Update path
                h_delta = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1-c1)**2) < (1.4 + 2/(d+1)) * chiN else 0
                p_c = (1 - c1) * p_c + h_delta * np.sqrt(c1 * (2 - c1) * mueff) * (new_mean - old_mean) / sigma

                mean = new_mean

                # Update covariance
                c1_term = c1 * (np.outer(p_c, p_c) - C)
                cmu_term = np.zeros((d, d))
                for i in range(mu):
                    z_i = eigenvectors.T @ (offspring[i] - old_mean) / sigma
                    cmu_term += weights[i] * np.outer(z_i, z_i)
                cmu_term = cmu * cmu_term

                C = (1 - c1 - cmu) * C + c1_term + cmu_term
                C = (C + C.T) / 2

                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-10)

                # Update sigma
                mean_shift = (new_mean - old_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (np.linalg.norm(mean_shift) - chiN)

                sigma *= np.exp(0.3 * cs / c_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

            return mean

        # Phase 1: CMA-ES for coarse search
        x_init = np.zeros(self.dim)
        x_init = cma_es_init(x_init, n_iter=30)

        # Phase 2: Multi-start Nelder-Mead
        # Start from CMA-ES result and some random points
        starts = [x_init.copy()]
        for i in range(5):
            if check_time() or num_calls >= self.budget:
                break
            x_rand = rng.uniform(-4, 4, size=self.dim)
            starts.append(x_rand)

        for x_start in starts:
            if check_time() or num_calls >= self.budget:
                break
            nelder_mead(x_start, max_iter=100)

        # Phase 3: Final polish with Nelder-Mead from best
        if not check_time() and num_calls < self.budget:
            nelder_mead(best_x.copy(), max_iter=50)

        return best_x, best_f
# EVOLVE-BLOCK-END
