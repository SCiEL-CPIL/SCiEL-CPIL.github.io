import numpy as np
import time
import math
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: CMA-ES for global search + Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0

        def time_left():
            return deadline - time.time()

        def do_eval(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- CMA-ES implementation ---
        def run_cmaes(mean_init, sigma_init, max_evals):
            nonlocal eval_count, best_x, best_f
            if time_left() < 5.0 or max_evals <= 0:
                return

            mean = np.clip(mean_init.copy(), -5, 5)
            sigma = sigma_init
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            local_evals = 0
            no_improve = 0
            best_local_f = best_f

            # Evaluate initial mean
            do_eval(mean)
            local_evals += 1

            # CMA-ES parameters
            lam = max(4, int(4 + 2 * math.log(dim)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()
            mueff = 1.0 / weights.sum()

            c1 = (2 * (mueff - 2 + 2 / dim)) / ((dim + 2) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mueff)
            cs = (lam / 2 + 1) / (dim + lam / 2 + 3)
            ds = 1 + 2 * max(0, math.sqrt((lam / dim - 1) / (dim + 1)) - 1) + cs

            while local_evals < max_evals and time_left() > 5.0:
                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * z
                    x = np.clip(x, -5, 5)
                    offspring.append(x)

                # Evaluate
                values = []
                for x in offspring:
                    if local_evals >= max_evals or time_left() < 2.0:
                        break
                    fx = do_eval(x)
                    values.append(fx)
                    local_evals += 1

                if len(values) < mu:
                    break

                # Sort
                idx = np.argsort(values)
                x_sorted = [offspring[i] for i in idx]
                f_sorted = [values[i] for i in idx]

                # Check for improvement
                if f_sorted[0] < best_local_f - 1e-10:
                    best_local_f = f_sorted[0]
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (x_sorted[i] - mean) / sigma
                old_mean = mean.copy()
                mean = mean + sigma * y_w
                mean = np.clip(mean, -5, 5)

                # Update step size
                y_sigma = y_w
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * y_sigma
                chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (local_evals / (cs * (dim + 1))))) < \
                       (1.4 + 2 / (dim + 1)) * chi_n
                sigma = sigma * math.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-15, 50.0)

                # Update covariance
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                pc = (1 - cmu) * pc + delta_hsig * (y_w + delta_hsig * ps)
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (x_sorted[i] - old_mean) / sigma
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * (2 - c1 - delta_hsig) + cmu) * C - c1 * rank1 - cmu * rankmu

                # Ensure PD
                eigvals, eigvecs = np.linalg.eigh(C)
                min_eig = np.min(eigvals)
                if min_eig < 1e-12:
                    C = C + (1e-12 - min_eig) * np.eye(dim)
                    eigvals, eigvecs = np.linalg.eigh(C)
                C = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T

                # Convergence check
                if sigma < 1e-12 or no_improve > 50:
                    break

        # --- Nelder-Mead local refinement ---
        def run_nelder_mead(x0, max_evals):
            nonlocal eval_count, best_x, best_f
            if time_left() < 3.0 or max_evals <= 0:
                return

            nm_x = x0.copy()
            nm_f = best_f

            # Initial simplex
            simplex = [nm_x.copy()]
            for i in range(dim):
                pt = nm_x.copy()
                step = 0.1 if nm_x[i] == 0 else 0.1 * abs(nm_x[i])
                pt[i] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            simplex_f = [nm_f]
            for i in range(1, dim + 1):
                simplex_f.append(do_eval(simplex[i]))

            max_nm_iter = min(200, (budget - eval_count) // (dim + 1))
            for _ in range(max_nm_iter):
                if eval_count >= budget or time_left() < 2.0:
                    break
                # Sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = do_eval(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = do_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = do_eval(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = do_eval(simplex[i])

        # --- Strategy: multiple phases ---
        # Phase 1: Broad exploration with CMA-ES
        run_cmaes(np.zeros(dim), 2.5, min(budget, 150))

        # Phase 2: Medium exploration around best found
        if time_left() > 10.0 and eval_count < budget:
            run_cmaes(best_x.copy(), 1.0, min(budget - eval_count, 100))

        # Phase 3: Finer refinement
        if time_left() > 10.0 and eval_count < budget:
            run_cmaes(best_x.copy(), 0.3, min(budget - eval_count, 80))

        # Phase 4: Very fine CMA-ES
        if time_left() > 5.0 and eval_count < budget:
            run_cmaes(best_x.copy(), 0.05, min(budget - eval_count, 50))

        # Phase 5: Nelder-Mead local search
        if time_left() > 3.0 and eval_count < budget:
            run_nelder_mead(best_x.copy(), min(budget - eval_count, 100))

        # Final phase: additional Nelder-Mead if budget remains
        if time_left() > 3.0 and eval_count < budget:
            run_nelder_mead(best_x.copy(), min(budget - eval_count, 50))

        return best_x, best_f
