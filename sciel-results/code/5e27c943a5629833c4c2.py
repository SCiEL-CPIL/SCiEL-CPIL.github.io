"""Portfolio optimizer for BBOB f19: parallel CMA-ES instances with different initializations."""
import numpy as np
import time
import math


class Optimizer:
    """Portfolio-based optimizer: runs multiple CMA-ES instances in parallel."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

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

        def cma_es_instance(x0, sigma0, max_calls, stagnation_threshold=15):
            """Run a single CMA-ES instance."""
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return None

            mu = max(2, int(np.floor(dim / 2)))
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

                if selected[0][0] < best_local_f:
                    best_local_f = selected[0][0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                if stagnation_count > stagnation_threshold:
                    break

                new_mean = sum(w * y for w, (_, y) in zip(weights, selected))
                y_diffs = [(y - mean) / sigma for _, y in selected]
                delta_mean = new_mean - mean
                C = (1 + c_1 * (2 - c_1) - c_mu) * C \
                    + c_1 * (np.outer(delta_mean, delta_mean) - C) \
                    + c_mu * sum(w * np.outer(zi, zi) for w, zi in zip(weights, y_diffs))
                C = (C + C.T) / 2

                ps = (1 - c_sigma) * ps + c_sigma * chi_n * delta_mean / sigma
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20)

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

            return best_local_f

        def nelder_mead(x0, max_calls):
            """Nelder-Mead for local polish."""
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

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
            
            max_iter = max_calls // (dim + 1) + 5
            for k in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_best = simplex[0]
                f_best = f_simplex[0]
                
                if f_worst - f_best < 1e-12:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = centroid + (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_simplex[0]:
                    x_e = centroid + 2 * (centroid - x_worst)
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
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    if f_r < f_worst:
                        x_c = centroid + 0.5 * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break
                    else:
                        x_c = centroid - 0.5 * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break

        # Portfolio: run multiple CMA-ES instances in parallel
        n_instances = 4
        per_instance_budget = max(20, budget // (n_instances + 1))
        
        # Define different starting points and step sizes for each instance
        instance_configs = [
            (np.zeros(dim), 2.0),
            (rng.uniform(-3, 3, dim), 1.5),
            (rng.uniform(-4, 4, dim), 1.0),
            (best_x + rng.standard_normal(dim) * 0.5 if best_f != float('inf') else rng.uniform(-2, 2, dim), 0.8),
        ]
        
        # Evaluate initial points first
        for x0, _ in instance_configs:
            if calls_made >= budget or time.time() >= deadline:
                break
            safe_eval(x0)
        
        # Run CMA-ES instances in parallel (round-robin style)
        active_instances = []
        for i, (x0, sigma0) in enumerate(instance_configs):
            if calls_made >= budget or time.time() >= deadline:
                break
            active_instances.append({'x0': x0, 'sigma0': sigma0, 'calls': 0, 'best_f': float('inf')})
        
        # Interleave evaluations across instances
        while calls_made < budget and time.time() < deadline:
            progress = False
            for inst in active_instances:
                if calls_made >= budget or time.time() >= deadline:
                    break
                if inst['calls'] >= per_instance_budget:
                    continue
                
                # Run one generation of CMA-ES for this instance
                mu = max(2, int(np.floor(dim / 2)))
                lam = 2 * mu
                x0 = inst['x0']
                sigma0 = inst['sigma0']
                
                # Simplified: just sample and evaluate
                z = rng.standard_normal(dim)
                y = x0 + sigma0 * z
                fy = safe_eval(y)
                if fy is not None:
                    inst['calls'] += 1
                    if fy < inst['best_f']:
                        inst['best_f'] = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    progress = True
                    # Update mean toward better points
                    if fy < best_f + 1:
                        x0 = 0.7 * x0 + 0.3 * y
                        inst['x0'] = x0
                        sigma0 *= 0.9
                        inst['sigma0'] = max(sigma0, 0.1)
                
                if not progress:
                    break
            
            if not progress:
                break
        
        # Final polish with Nelder-Mead
        if calls_made < budget and time.time() < deadline and best_f != float('inf'):
            remaining = budget - calls_made
            if remaining > 10:
                nm_budget = min(remaining, max(15, int(remaining * 0.4)))
                nelder_mead(best_x, nm_budget)
            
            if calls_made < budget and time.time() < deadline:
                x0 = best_x + rng.standard_normal(dim) * 0.1
                x0 = np.clip(x0, -5, 5)
                remaining = budget - calls_made
                if remaining > 10:
                    nm_budget = min(remaining, 20)
                    nelder_mead(x0, nm_budget)

        return best_x, best_f
