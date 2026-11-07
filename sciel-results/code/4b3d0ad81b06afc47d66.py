"""Combined optimizer for BBOB f19: CMA-ES with adaptive restarts and Nelder-Mead polish."""
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES based optimizer with multi-start, adaptive restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(12345)

        # Nelder-Mead implementation for local polish
        def nelder_mead(x0, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_temp = x0.copy()
                if x_temp[i] == 0:
                    x_temp[i] = 0.05
                else:
                    x_temp[i] += 0.05
                simplex.append(x_temp)
            
            f_simplex = []
            for i in range(dim + 1):
                if calls_made >= budget or time.time() >= deadline:
                    return
                fx = safe_eval(simplex[i])
                if fx is None:
                    return
                f_simplex.append(fx)
            
            # Main loop
            max_iter = max_calls // (dim + 1) + 5
            for k in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_best = simplex[0]
                f_best = f_simplex[0]
                
                # Check for convergence
                if f_worst - f_best < 1e-12:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break

            return

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_calls, stagnation_threshold=15):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            dim = self.dim
            mu = max(2, int(np.floor(dim / 2)))  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            mean = np.clip(x0.copy(), -5, 5)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            best_local_f = float('inf')
            stagnation_count = 0
            lam = 2 * mu

            for gen in range(max_calls // mu + 10):
                if calls_made >= budget or time.time() >= deadline:
                    break

                # Sample lambda points
                offspring = []
                for i in range(lam):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (z @ np.linalg.cholesky(C))
                    fy = safe_eval(y)
                    if fy is not None:
                        offspring.append((fy, y))

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]

                # Track local best
                if selected[0][0] < best_local_f:
                    best_local_f = selected[0][0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Adaptive restart: if stagnation detected, break to allow restart
                if stagnation_count > stagnation_threshold:
                    break

                # Update mean
                new_mean = sum(w * y for w, (_, y) in zip(weights, selected))

                # Update covariance
                y_diffs = [(y - mean) / sigma for _, y in selected]
                delta_mean = new_mean - mean
                C = (1 + c_1 * (2 - c_1) - c_mu) * C \
                    + c_1 * (np.outer(delta_mean, delta_mean) - C) \
                    + c_mu * sum(w * np.outer(zi, zi) for w, zi in zip(weights, y_diffs))
                C = (C + C.T) / 2

                # Update step size
                ps = (1 - c_sigma) * ps + c_sigma * chi_n * delta_mean / sigma
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20)

                # Update evolution path
                pc = (1 - c_c) * pc + c_c * (delta_mean / sigma)
                cs = math.exp(2 - c_c)
                C = (1 + c_1 * (2 - c_1) - c_mu - c_c * (1 - c_c) * (2 - c_c)) * C \
                    + c_1 * np.outer(delta_mean, delta_mean) \
                    + c_mu * sum(w * np.outer(zi, zi) for w, zi in zip(weights, y_diffs)) \
                    + c_c * (1 - c_c) * (cs * np.outer(pc, pc) - C)
                C = (C + C.T) / 2

                mean = new_mean
                if np.linalg.norm(mean) > 100:
                    break

            return mean, best_local_f

        # Phase 1: Initial exploration with diverse starting points
        n_starts = 8
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Evaluate starting points
        for i, x0 in enumerate(starts):
            if calls_made >= budget or time.time() >= deadline:
                break
            fx = safe_eval(x0)
            if fx is None:
                break

        # Phase 2: CMA-ES with adaptive restarts
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            # First CMA-ES run with larger budget
            cma_budget = min(remaining, max(50, int(remaining * 0.6)))
            cma_es(best_x, 1.5, cma_budget, stagnation_threshold=12)
            
            # If budget remains, do a second CMA-ES from a perturbed best
            if calls_made < budget and time.time() < deadline:
                x0 = best_x + rng.standard_normal(dim) * 0.3
                x0 = np.clip(x0, -5, 5)
                remaining = budget - calls_made
                if remaining > 20:
                    cma_budget = min(remaining, max(30, int(remaining * 0.4)))
                    cma_es(x0, 0.8, cma_budget, stagnation_threshold=10)
            
            # If budget remains, try a fresh random start
            if calls_made < budget and time.time() < deadline:
                x0 = rng.uniform(-4, 4, dim)
                remaining = budget - calls_made
                if remaining > 20:
                    cma_budget = min(remaining, 40)
                    cma_es(x0, 1.0, cma_budget, stagnation_threshold=8)

        # Phase 3: Nelder-Mead polish on the best point found
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 10:
                nm_budget = min(remaining, max(15, int(remaining * 0.3)))
                nelder_mead(best_x, nm_budget)
            
            # One more NM from a slightly perturbed best if budget allows
            if calls_made < budget and time.time() < deadline:
                x0 = best_x + rng.standard_normal(dim) * 0.1
                x0 = np.clip(x0, -5, 5)
                remaining = budget - calls_made
                if remaining > 10:
                    nm_budget = min(remaining, 20)
                    nelder_mead(x0, nm_budget)

        return best_x, best_f
