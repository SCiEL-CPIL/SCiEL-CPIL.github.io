"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Optimizer: Population-based ES for exploration, 
    (1+1)-ES with Rastrigin-aware jumps for exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Population-based Exploration ---
        # Use a (mu+lambda) ES to find a promising basin.
        mu = 5
        lam = 15
        sigma = 2.0
        
        # Initialize population
        pop = np.array([np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(mu)])
        pop_f = np.array([eval_func(p) for p in pop])
        
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]
        
        gen = 0
        # Run for ~40% of budget
        max_gen_exploration = int(budget * 0.4 / lam)
        
        while gen < max_gen_exploration and call_count < budget and time.time() < deadline:
            gen += 1
            offspring = []
            offspring_f = []
            
            for _ in range(lam):
                if call_count >= budget or time.time() > deadline:
                    break
                # Tournament selection from parents
                p1_idx = rng.integers(0, mu)
                p2_idx = rng.integers(0, mu)
                parent = pop[p1_idx] if pop_f[p1_idx] < pop_f[p2_idx] else pop[p2_idx]
                
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                child_f = eval_func(child)
                offspring.append(child)
                offspring_f.append(child_f)
            
            if not offspring:
                break
                
            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)
            
            # Combine and select
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([pop_f, offspring_f])
            idx = np.argsort(all_f)[:mu]
            pop = all_x[idx]
            pop_f = all_f[idx]
            
            # Adapt sigma (1/5 rule approximation)
            # If we found a better solution, increase sigma, else decrease
            if best_f < pop_f[0] + 1e-6: 
                 # This check is tricky with nonlocal, let's use a simpler heuristic
                 pass
            
            # Simple adaptation: if population is diverse, keep sigma, else reduce
            # Check variance of the population
            if np.var(pop, axis=0).mean() < 0.1:
                sigma *= 0.8
            else:
                sigma *= 1.05
            sigma = np.clip(sigma, 0.1, 5.0)

        # --- Phase 2: (1+1)-ES with Rastrigin-aware Jumps ---
        # Start from the best point found in Phase 1
        x_cur = best_x.copy() if best_x is not None else pop[0].copy()
        f_cur = best_f if best_x is not None else pop_f[0]
        
        sigma_exploit = 0.5
        # Run for remaining budget
        while call_count < budget and time.time() < deadline:
            # Generate standard Gaussian step
            x_gauss = np.clip(x_cur + sigma_exploit * rng.standard_normal(dim), -5, 5)
            f_gauss = eval_func(x_gauss)
            
            best_candidate_x = x_cur
            best_candidate_f = f_cur
            
            if f_gauss < best_candidate_f:
                best_candidate_x = x_gauss
                best_candidate_f = f_gauss

            # Generate Rastrigin-aware jumps: shift each dimension by +/- 2*pi
            # This helps escape local minima in the rotated landscape
            for i in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Try + 2*pi
                x_jump_plus = x_cur.copy()
                x_jump_plus[i] = np.clip(x_cur[i] + 2.0 * np.pi, -5, 5)
                f_jump_plus = eval_func(x_jump_plus)
                if f_jump_plus < best_candidate_f:
                    best_candidate_x = x_jump_plus
                    best_candidate_f = f_jump_plus

                # Try - 2*pi
                x_jump_minus = x_cur.copy()
                x_jump_minus[i] = np.clip(x_cur[i] - 2.0 * np.pi, -5, 5)
                f_jump_minus = eval_func(x_jump_minus)
                if f_jump_minus < best_candidate_f:
                    best_candidate_x = x_jump_minus
                    best_candidate_f = f_jump_minus
            
            # Update current point if improved
            if best_candidate_f < f_cur:
                x_cur = best_candidate_x
                f_cur = best_candidate_f
                # Increase sigma slightly if we made a big jump? 
                # Or just keep it. Let's adapt sigma based on success.
                sigma_exploit = min(sigma_exploit * 1.1, 2.0)
            else:
                # No improvement, decrease sigma
                sigma_exploit = max(sigma_exploit * 0.9, 0.05)
            
            # If sigma is very small, try a random restart in the neighborhood
            if sigma_exploit < 0.1 and call_count < budget - 10:
                # Small random perturbation to escape
                x_cur = np.clip(x_cur + 0.5 * rng.standard_normal(dim), -5, 5)
                f_cur = eval_func(x_cur)
                sigma_exploit = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
