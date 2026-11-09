"""Direct optimizer exploiting Rastrigin periodicity via multi-start CMA-ES with periodicity-aware restarts."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer using multi-start CMA-ES with adaptive restarts for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
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

        # Phase 1: Initial sampling - mix of random and structured points
        # Rastrigin has period 2*pi, so try offsets near common minima locations
        n_random = 15
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        # Try structured points: near 0, near +/-pi/2, etc.
        structured_candidates = [
            np.zeros(dim),
            np.full(dim, np.pi/2),
            np.full(dim, -np.pi/2),
            np.full(dim, np.pi),
            np.full(dim, -np.pi),
        ]
        for sc in structured_candidates:
            if not time_ok():
                break
            # Add small random perturbation to avoid exact duplicates
            x = sc + rng.standard_normal(dim) * 0.1
            x = np.clip(x, -5, 5)
            evaluate(x)
        
        if not time_ok():
            return (best_x, best_f)

        # CMA-ES parameters
        lambda_pop = 12
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
                
                if no_improve > 25:
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
                sigma = float(np.clip(sigma, 0.001, 10.0))
                
                x_mean = x_new_mean
                
                if sigma < 0.001:
                    break
            
            return best_f

        # Phase 2: Multi-start CMA-ES with Rastrigin-aware restarts
        # Rastrigin has period 2*pi, so minima repeat at x + 2*pi*k
        # Strategy: run CMA-ES from best, then try periodic offsets
        
        remaining_budget = budget - calls_used
        
        # Define restart points: best + periodic offsets
        # In 5D, we try shifting each coordinate by +/-pi (half period)
        # This is cheaper than full 2*pi shifts and still exploits periodicity
        def get_restart_candidates(current_best):
            """Generate restart candidates exploiting Rastrigin periodicity."""
            candidates = [current_best.copy()]
            
            # Try shifts of +/- pi in random subsets of dimensions
            for trial in range(8):
                candidate = current_best.copy()
                # Pick 1-2 random dimensions to shift
                num_shifts = rng.integers(1, 3)
                dims_to_shift = rng.choice(dim, size=num_shifts, replace=False)
                for d in dims_to_shift:
                    shift = rng.choice([-1, 1]) * math.pi
                    candidate[d] += shift
                candidate = np.clip(candidate, -5, 5)
                candidates.append(candidate)
            
            # Also try a few fully random points
            for _ in range(3):
                candidates.append(rng.uniform(-5, 5, dim))
            
            return candidates
        
        # Phase 2a: Primary CMA-ES run from best point
        primary_evals = min(int(remaining_budget * 0.4), 120)
        if primary_evals > 0 and time_ok():
            run_cmaes(best_x.copy(), 1.0, primary_evals)
        
        # Phase 2b: Secondary CMA-ES runs from periodic restarts
        remaining_after_primary = budget - calls_used
        if remaining_after_primary > 20 and time_ok():
            restarts = get_restart_candidates(best_x)
            # Sort by distance to best (prefer closer ones first)
            restarts.sort(key=lambda x: np.linalg.norm(x - best_x))
            
            # Run CMA-ES from up to 4 best restart candidates
            num_restarts = min(4, len(restarts))
            evals_per_restart = max(10, int(remaining_after_primary * 0.15))
            
            for i in range(num_restarts):
                if not time_ok():
                    break
                x0 = restarts[i]
                sigma0 = 0.8 if i > 0 else 1.0
                max_evals = min(evals_per_restart, budget - calls_used)
                if max_evals > 0:
                    run_cmaes(x0, sigma0, max_evals)
        
        # Phase 3: Final refinement with small sigma CMA-ES
        if time_ok() and budget - calls_used > 10:
            max_evals = min(40, budget - calls_used)
            run_cmaes(best_x.copy(), 0.05, max_evals)

        return (best_x, best_f)
