"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restart mechanism and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # CMA-ES Parameters
        lambda_pop = 9
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)
        
        cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
        cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
        c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))
        
        def run_cmaes(x_init, sigma_init, max_calls):
            nonlocal best_x, best_f, calls_made
            
            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_x = x_mean.copy()
            local_best_f = func(x_mean)
            calls_made += 1
            
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            last_improvement = local_best_f
            no_improve_gen = 0
            gen = 0
            
            while calls_made < max_calls:
                if time.time() >= deadline:
                    break
                
                # Sample lambda points
                offspring = []
                for i in range(lambda_pop):
                    if calls_made >= max_calls:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls_made += 1
                    offspring.append((x_new, f_new))
                
                if len(offspring) == 0:
                    break
                    
                # Sort by fitness
                offspring.sort(key=lambda x: x[1])
                
                # Update mean
                x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0)
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (B.T @ (x_new_mean - x_mean))))
                
                hsig = (np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff))
                
                if hsig:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))
                
                # Update covariance
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)
                    
                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                
                # Update mean
                x_mean = x_new_mean
                gen += 1
                
                # Update best
                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                    last_improvement = local_best_f
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1
                
                # Update basis
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-10))
                except:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-10))
                
                # Check for stall - restart if no improvement for 12 generations
                if no_improve_gen >= 12 and calls_made < max_calls - 20:
                    break
                
                # Check convergence
                if np.linalg.norm(D) < 1e-10:
                    break
            
            return local_best_f, local_best_x, calls_made
        
        # Multi-start strategy: try up to 5 different starting points
        # Allocate budget more aggressively to early starts to capture early progress
        num_starts = 5
        # Budget allocation: 40%, 25%, 15%, 10%, 10%
        budget_fracs = [0.40, 0.25, 0.15, 0.10, 0.10]
        
        for start_idx in range(num_starts):
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break
            
            # Allocate budget for this start
            alloc = int(total_budget * budget_fracs[start_idx])
            max_calls_for_this = min(calls_made + alloc, total_budget)
            
            # Different initialization for each start
            if start_idx == 0:
                x_init = rng.uniform(-5, 5, dim)
                sigma_init = 2.5
            elif start_idx == 1:
                # Start near the previous best if we have one
                if best_x is not None:
                    x_init = np.clip(best_x + rng.normal(0, 1.5, dim), -5, 5)
                else:
                    x_init = rng.uniform(-5, 5, dim)
                sigma_init = 1.5
            elif start_idx == 2:
                # Random restart
                x_init = rng.uniform(-5, 5, dim)
                sigma_init = 2.0
            elif start_idx == 3:
                # Start near previous best with smaller perturbation
                if best_x is not None:
                    x_init = np.clip(best_x + rng.normal(0, 0.8, dim), -5, 5)
                else:
                    x_init = rng.uniform(-5, 5, dim)
                sigma_init = 1.0
            else:
                # Random restart far from current best
                if best_x is not None:
                    direction = best_x / (np.linalg.norm(best_x) + 1e-10)
                    x_init = np.clip(-3 * direction + rng.normal(0, 1.5, dim), -5, 5)
                else:
                    x_init = rng.uniform(-5, 5, dim)
                sigma_init = 2.5
            
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls_for_this)
            
            # If we still have budget and haven't hit deadline, continue
            if calls_made >= total_budget or time.time() >= deadline:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
