"""Direct optimizer exploiting Rastrigin periodicity via hierarchical adaptive ES with refinement."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer using hierarchical budget allocation with adaptive ES for Rastrigin."""

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

        # Phase 1: Broad initial exploration with mixed strategies
        # Use 30 evaluations for diverse initial sampling
        n_random = 30
        for i in range(n_random):
            if not time_ok():
                break
            # Mix of uniform, biased toward center, and structured starts
            if i < 10:
                x = rng.uniform(-5, 5, dim)
            elif i < 20:
                x = rng.uniform(-3, 3, dim)
            else:
                # Structured grid-like starts
                x = np.zeros(dim)
                for d in range(dim):
                    if (i + d) % 2 == 0:
                        x[d] = rng.uniform(-5, 5)
                    else:
                        x[d] = rng.uniform(-2, 2)
            evaluate(x)

        if not time_ok():
            return (best_x, best_f)

        # Phase 2: Hierarchical adaptive ES with 3 offspring per generation
        # Allocate budget hierarchically: top candidate gets 1/4, next 1/8, etc.
        
        def run_adaptive_es(x0, max_evals, sigma_init=1.0, n_offspring=3):
            """Run adaptive ES with n_offspring offspring per generation."""
            nonlocal calls_used, best_x, best_f
            
            x_mean = x0.copy()
            sigma = sigma_init
            C = np.eye(dim)
            evals_this_run = 0
            no_improve = 0
            last_best = best_f
            
            # Adaptive parameters
            c_sigma = 1.0 / (dim + 1)
            c_c = 1.0 / (dim + 3)
            c_1 = 2 / ((dim + 1.5) ** 2 + 1)
            c_mu = min(1 - c_1, 1.0 / (dim + 1))
            chi_N = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            evolution_path_sigma = np.zeros(dim)
            evolution_path_cov = np.zeros((dim, dim))
            
            while evals_this_run < max_evals and time_ok():
                # Generate offspring
                offspring = []
                for _ in range(n_offspring):
                    if not time_ok() or evals_this_run >= max_evals:
                        break
                    z = rng.standard_normal(dim)
                    x_off = x_mean + sigma * (C @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = evaluate(x_off)
                    evals_this_run += 1
                    offspring.append((fx_off, x_off))
                
                if len(offspring) < 1:
                    break
                
                offspring.sort(key=lambda t: t[0])
                best_off = offspring[0]
                
                if best_off[0] < last_best - 1e-10:
                    last_best = best_off[0]
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 20:
                    break
                
                # Update mean toward best offspring
                x_new_mean = best_off[1].copy()
                
                # Update sigma evolution path
                evolution_path_sigma = (1 - c_sigma) * evolution_path_sigma + \
                                       math.sqrt(c_sigma * (2 - c_sigma)) * \
                                       (C @ ((x_new_mean - x_mean) / sigma))
                
                h_sigma = float(np.linalg.norm(evolution_path_sigma) / chi_N < (1.4 + 2 / (dim + 1)))
                
                # Update covariance evolution path
                evolution_path_cov = (1 - c_c) * evolution_path_cov + \
                                     h_sigma * c_c * (1 - c_c) * np.outer(x_new_mean - x_mean,
                                                                          x_new_mean - x_mean)
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * evolution_path_cov + c_mu * rank_one
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= math.exp(c_sigma / chi_N * (np.linalg.norm(evolution_path_sigma) / chi_N - 1))
                sigma = float(np.clip(sigma, 0.001, 10.0))
                
                x_mean = x_new_mean
                
                if sigma < 0.001:
                    break
            
            return best_f

        # Phase 3: Hierarchical budget allocation
        remaining = budget - calls_used
        if remaining <= 0:
            return (best_x, best_f)
        
        # Allocate budget: top 25%, next 12.5%, next 6.25%, etc.
        # Use up to 4 runs with decreasing budget
        allocations = [0.25, 0.125, 0.0625, 0.03125]
        
        # Start from best point found in Phase 1
        current_best_x = best_x.copy()
        
        for alloc in allocations:
            if not time_ok():
                break
            
            max_evals = int(remaining * alloc)
            if max_evals < 10:
                break
            
            # Start from current best with moderate sigma
            run_adaptive_es(current_best_x, max_evals, sigma_init=0.5, n_offspring=3)
            
            # Update current best
            current_best_x = best_x.copy()
        
        # Phase 4: Final refinement with small sigma
        if time_ok() and budget - calls_used > 20:
            max_evals = min(50, budget - calls_used)
            run_adaptive_es(best_x.copy(), max_evals, sigma_init=0.05, n_offspring=3)

        return (best_x, best_f)
