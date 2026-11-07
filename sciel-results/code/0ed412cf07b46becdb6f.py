"""
Multi-restart CMA-ES optimizer for BBOB f3 (Rastrigin).
Uses multiple independent CMA-ES runs with different initializations to
handle the multimodality of Rastrigin. Each restart gets a portion of
the budget, and the best result across all restarts is returned.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def run_cmaes(initial_mean, initial_sigma, max_calls, local_rng):
            """Run CMA-ES with given initial conditions, using up to max_calls evaluations."""
            nonlocal best_x, best_f, calls_made
            
            # CMA-ES Parameters
            pop_size = 4 + int(3 * np.log(n))
            if pop_size > max_calls:
                pop_size = max(2, max_calls // 2)
            
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            mean = initial_mean.copy()
            sigma = initial_sigma
            
            C = np.eye(n)
            C_sqrt = np.eye(n)
            
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))
            
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            no_improve = 0
            last_best = float('inf')
            
            while not check_deadline():
                # Generate offspring
                offspring = []
                for i in range(pop_size):
                    z = local_rng.standard_normal(n)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    update_best(y, fy)
                    offspring.append((fy, y))
                    
                    if check_deadline():
                        break
                
                if len(offspring) < mu:
                    break
                    
                # Sort offspring
                offspring.sort(key=lambda x: x[0])
                
                # Check improvement
                if offspring[0][0] < last_best - 1e-10:
                    last_best = offspring[0][0]
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]
                    
                # Update mean
                old_mean = mean.copy()
                mean = y_w
                
                # Update covariance matrix
                ys = (y_w - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n < (1.4 + 2 / (n + 1)) * cc)
                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                else:
                    pc = (1 - cs) * pc
                
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                    
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * C) + cmu * rank_mu
                
                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2.0
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                
                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Convergence criteria
                if sigma < 1e-12:
                    break
                if no_improve > 20 and sigma < 0.3:
                    break

        # Multi-restart strategy:
        # First restart: large sigma, centered at origin (global exploration)
        # Subsequent restarts: smaller sigma, centered at best found so far (local refinement)
        
        # Restart 1: Global exploration with large sigma
        initial_mean1 = np.zeros(n)
        initial_sigma1 = 3.0
        max_calls1 = int(budget * 0.4)
        run_cmaes(initial_mean1, initial_sigma1, max_calls1, rng)
        
        if check_deadline():
            return best_x, best_f
        
        # Restart 2: Local refinement around best found, medium sigma
        if best_x is not None:
            initial_mean2 = best_x.copy()
            initial_sigma2 = 0.5
            max_calls2 = int(budget * 0.3)
            run_cmaes(initial_mean2, initial_sigma2, max_calls2, rng)
        
        if check_deadline():
            return best_x, best_f
        
        # Restart 3: Another local refinement with different center (perturbed best)
        if best_x is not None:
            perturbation = rng.uniform(-0.5, 0.5, size=n)
            initial_mean3 = np.clip(best_x + perturbation, -5, 5)
            initial_sigma3 = 0.3
            max_calls3 = budget - calls_made
            if max_calls3 > 0:
                run_cmaes(initial_mean3, initial_sigma3, max_calls3, rng)
        
        # Final phase: Use remaining budget for fine-tuning around best
        if best_x is not None and not check_deadline():
            initial_mean4 = best_x.copy()
            initial_sigma4 = 0.1
            max_calls4 = budget - calls_made
            if max_calls4 > 0:
                run_cmaes(initial_mean4, initial_sigma4, max_calls4, rng)

        return best_x, best_f

# EVOLVE-BLOCK-END
