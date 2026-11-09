"""Direct optimizer exploiting Rastrigin periodicity via lattice search + CMA-ES refinement."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer exploiting Rastrigin periodicity + CMA-ES for rotation-aware search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # Phase 1: Dense lattice search exploiting Rastrigin's periodicity
        # The Rastrigin function has local minima at integer points in the rotated frame.
        # We search on a lattice with spacing 1.0 (the period) and also half-period 0.5.
        # This should find points close to the true minimum efficiently.
        
        # First, try a coarse grid with step 1.0
        # The search space is [-5, 5]^5, so with step 1.0 we have 11^5 = 161051 points.
        # That's too many. Let's be smarter: use a smaller region or adaptive approach.
        
        # Actually, let's use a hierarchical approach:
        # 1. Try a few random points to get an initial estimate
        # 2. Then do local lattice search around the best point
        
        # Initial random sampling
        n_random = 30
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        if not time_ok():
            return (best_x, best_f)
        
        # Phase 2: Local lattice search around the current best
        # Try offsets of 0, ±0.5, ±1.0, ±1.5, ±2.0 in each dimension
        # This is a local search that exploits the periodicity
        
        offsets = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])
        offset_list = list(np.meshgrid(*([offsets] * dim), indexing='ij'))
        
        # Sort offsets by distance from center to prioritize closer points
        center = best_x.copy()
        for i in range(len(offset_list[0].flat)):
            if not time_ok():
                break
            x = center + np.array([offset_list[d].flat[i] for d in range(dim)])
            if np.all(x >= -5) and np.all(x <= 5):
                evaluate(x)
        
        if not time_ok():
            return (best_x, best_f)
        
        # Phase 3: CMA-ES refinement from the best point found
        # Use a well-tuned CMA-ES with adaptive restarts
        
        lambda_pop = 10 + int(3 * math.log(dim))
        mu = lambda_pop // 2
        
        c_sigma = (mu + 0.5) / (dim + 2) / 5
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        chi_N = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        w /= w.sum()
        nu_eff = 1 / sum(w ** 2)
        
        def run_cmaes(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0 for at most max_evals evaluations."""
            nonlocal calls_used, best_x, best_f
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            evolution_path_sigma = np.zeros(dim)
            evolution_path_cov = np.zeros((dim, dim))
            
            last_best = best_f
            no_improve = 0
            evals_this_run = 0
            
            while evals_this_run < max_evals and time_ok():
                # Generate population
                offspring = []
                for _ in range(lambda_pop):
                    if not time_ok() or evals_this_run >= max_evals:
                        break
                    z = rng.standard_normal(dim)
                    x_off = x_mean + sigma * (C @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = evaluate(x_off)
                    evals_this_run += 1
                    offspring.append((fx_off, x_off))
                
                if len(offspring) < mu:
                    break
                
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]
                
                if selected[0][0] < last_best - 1e-10:
                    last_best = selected[0][0]
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 15:
                    break
                
                # Update mean
                x_new_mean = sum(w[i] * selected[i][1] for i in range(mu))
                
                # Update sigma evolution path
                evolution_path_sigma = (1 - c_sigma) * evolution_path_sigma + \
                                       math.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * \
                                       (C @ ((x_new_mean - x_mean) / sigma))
                
                h_sigma = float(np.linalg.norm(evolution_path_sigma) / chi_N < (1.4 + 2 / (dim + 1)))
                
                # Update covariance evolution path
                evolution_path_cov = (1 - c_c) * evolution_path_cov + \
                                     h_sigma * c_c * (1 - c_c) * np.outer(x_new_mean - x_mean,
                                                                          x_new_mean - x_mean)
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
                rank_mu = sum(w[i] * np.outer(selected[i][1] - x_mean,
                                              selected[i][1] - x_mean) for i in range(mu))
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * evolution_path_cov + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= math.exp(c_sigma / chi_N * (np.linalg.norm(evolution_path_sigma) / chi_N - 1))
                sigma = float(np.clip(sigma, 0.005, 10.0))
                
                x_mean = x_new_mean
                
                if sigma < 0.005:
                    break
            
            return best_f
        
        # Run CMA-ES from the best point with a few different initial sigmas
        sigmas_to_try = [0.5, 0.2, 0.1, 0.05]
        for sigma0 in sigmas_to_try:
            if not time_ok():
                break
            max_evals = min(80, budget - calls_used)
            if max_evals > 0:
                run_cmaes(best_x.copy(), sigma0, max_evals)
        
        return (best_x, best_f)
