import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = float(func(x))
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Single CMA-ES run with adaptive restarts (not independent restarts)
        # This preserves covariance structure across restarts
        popsize = 10  # Smaller for 5D to allow more generations
        mu = max(1, popsize // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        rng = np.random.default_rng(42)

        # Start at origin with large sigma to cover the box
        x = np.zeros(dim)
        sigma = 2.5
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        gen = 0

        try:
            # Initial population
            pop = []
            for i in range(popsize):
                if not time_ok() or calls >= self.budget:
                    break
                xi = x + sigma * rng.standard_normal(dim)
                xi = np.clip(xi, -5, 5)
                fi = wrapped_func(xi)
                pop.append((fi, xi))

            if len(pop) < popsize:
                # Fall through to local refinement with what we have
                pass
            else:
                pop.sort(key=lambda t: t[0])

                # Main CMA-ES loop with adaptive restart
                no_improve = 0
                last_best = best_f
                restart_threshold = 20  # Generations without improvement before restart

                while calls < self.budget and time_ok():
                    # Track improvement
                    if pop[0][0] < best_f:
                        best_f = pop[0][0]
                        best_x = pop[0][1].copy()
                    
                    if best_f < last_best - 1e-10:
                        no_improve = 0
                        last_best = best_f
                    else:
                        no_improve += 1

                    # Adaptive restart: if stuck, re-center near best with fresh covariance
                    if no_improve >= restart_threshold and gen > 5:
                        # Re-center at best point, reset covariance, moderate sigma
                        x = best_x.copy()
                        sigma = 0.5
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        no_improve = 0
                        gen = 0  # Reset generation counter for this "sub-run"
                        
                        # Generate new population around best
                        pop = []
                        for i in range(popsize):
                            if not time_ok() or calls >= self.budget:
                                break
                            xi = x + sigma * rng.standard_normal(dim)
                            xi = np.clip(xi, -5, 5)
                            fi = wrapped_func(xi)
                            pop.append((fi, xi))
                        
                        if len(pop) < popsize:
                            break
                        pop.sort(key=lambda t: t[0])
                        continue

                    # Mean update
                    x_old = x.copy()
                    x = sum(weights[i] * pop[i][1] for i in range(mu))

                    # Update sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (x - x_old))
                    norm_p_sigma = np.linalg.norm(p_sigma)
                    if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                        sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))

                    # Update covariance
                    diff_pop = [pop[i][1] - x_old for i in range(mu)]
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x - x_old) / sigma
                    norm_p_c = np.linalg.norm(p_c)
                    delta_heav = 1 if norm_p_c / np.sqrt(1 - (1 - c_c) ** (gen + 1)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                    C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(diff_pop[i], diff_pop[i]) for i in range(mu))

                    # Ensure C is positive definite
                    if gen % 3 == 0:
                        C = (C + C.T) / 2
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-12)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                    # Generate new population
                    new_pop = []
                    for i in range(popsize):
                        if not time_ok() or calls >= self.budget:
                            break
                        try:
                            L = np.linalg.cholesky(C)
                            xi = x + sigma * (L @ rng.standard_normal(dim))
                        except:
                            xi = x + sigma * rng.standard_normal(dim)
                        xi = np.clip(xi, -5, 5)
                        fi = wrapped_func(xi)
                        new_pop.append((fi, xi))

                    if len(new_pop) < popsize:
                        break

                    pop = new_pop
                    pop.sort(key=lambda t: t[0])
                    gen += 1

                    # Check for convergence
                    if sigma < 1e-10:
                        break

        except (TimeoutError, Exception):
            pass

        # Phase 2: Nelder-Mead refinement for final convergence
        remaining_budget = self.budget - calls
        
        if remaining_budget > 10 and best_x is not None and time_ok():
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            for i in range(dim):
                pert = best_x.copy()
                pert[i] += 0.1
                simplex.append(pert)
            
            # Evaluate initial simplex
            simplex_f = []
            for i, point in enumerate(simplex):
                if not time_ok() or calls >= self.budget:
                    break
                f = wrapped_func(point)
                simplex_f.append(f)
            
            if len(simplex_f) < dim + 1:
                pass
            else:
                # Nelder-Mead iterations
                alpha = 1.0  # Reflection
                gamma = 2.0  # Expansion
                rho = 0.5    # Contraction
                sigma_nm = 0.5  # Shrink
                
                for _ in range(remaining_budget - dim - 1):
                    if not time_ok() or calls >= self.budget:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflect
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = wrapped_func(x_r)
                    
                    if f_r < simplex_f[0]:
                        # Expand
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = wrapped_func(x_e)
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
                        # Contract
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = wrapped_func(x_c)
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                simplex_f[i] = wrapped_func(simplex[i])
                    
                    # Check convergence
                    if max(simplex_f) - min(simplex_f) < 1e-12:
                        break

        return best_x, best_f
