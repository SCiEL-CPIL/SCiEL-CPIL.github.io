"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using Multi-Start (1+8)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Solver using Multi-Start (1+8)-ES with per-dimension adaptive step sizes.
    
    Strategy:
    1. Initial exploration with random sampling to find promising basins.
    2. Run multiple (1+8)-ES instances from the best starting points.
    3. Each instance uses per-dimension step sizes that adapt based on success rate.
    4. This handles the ill-conditioning of Schaffers F7 better than isotropic ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        rng = np.random.default_rng(42)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Initial Exploration ---
        # Use 20% of budget for random sampling
        n_init = max(20, int(budget * 0.2))
        
        init_points = np.zeros((n_init, dim))
        for i in range(dim):
            init_points[:, i] = rng.uniform(-5, 5, n_init)
        
        init_scores = np.full(n_init, np.inf)
        for i in range(n_init):
            if time.time() > deadline:
                break
            init_scores[i] = counting_func(init_points[i])
        
        # Select top K points for multi-start
        K = 3
        valid_indices = np.where(init_scores < np.inf)[0]
        
        if len(valid_indices) == 0:
            # Fallback
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)
            return best_x, best_f
        
        sorted_indices = valid_indices[np.argsort(init_scores[valid_indices])]
        start_indices = sorted_indices[:K]
        
        # --- Phase 2: Multi-Start (1+8)-ES ---
        # Run (1+8)-ES from each start point, sharing budget
        
        # Number of offspring per generation
        lambda_offspring = 8
        
        for idx in start_indices:
            if calls >= budget or time.time() > deadline:
                break
            
            x_current = init_points[idx].copy()
            f_current = init_scores[idx]
            
            # Per-dimension step sizes
            sigma = np.ones(dim) * 0.5
            
            # Success counters per dimension for adaptive step size
            success_counts = np.zeros(dim)
            total_counts = np.zeros(dim)
            
            # Stagnation tracking
            stagnation = 0
            max_stagnation = 30
            
            while calls < budget - 10 and time.time() < deadline:
                # Generate offspring
                offspring = []
                offspring_f = []
                
                for j in range(lambda_offspring):
                    if calls >= budget - 10 or time.time() > deadline:
                        break
                    
                    # Sample candidate with per-dimension step sizes
                    z = rng.standard_normal(dim)
                    x_candidate = x_current + sigma * z
                    x_candidate = np.clip(x_candidate, -5, 5)
                    
                    f_candidate = counting_func(x_candidate)
                    offspring.append(x_candidate)
                    offspring_f.append(f_candidate)
                
                if len(offspring) == 0:
                    break
                
                # Select best offspring
                best_offspring_idx = np.argmin(offspring_f)
                best_offspring = offspring[best_offspring_idx]
                best_offspring_f = offspring_f[best_offspring_idx]
                
                # Update success counts per dimension
                for d in range(dim):
                    if best_offspring[d] > x_current[d]:
                        if best_offspring_f < f_current:
                            success_counts[d] += 1
                    elif best_offspring[d] < x_current[d]:
                        if best_offspring_f < f_current:
                            success_counts[d] += 1
                    total_counts[d] += 1
                
                # Adapt step sizes based on success rate (1/5 rule per dimension)
                for d in range(dim):
                    if total_counts[d] > 0:
                        success_rate = success_counts[d] / total_counts[d]
                        if success_rate > 0.2:
                            sigma[d] *= 1.3
                        elif success_rate < 0.2:
                            sigma[d] /= 1.3
                        
                        # Keep step sizes within reasonable bounds
                        sigma[d] = np.clip(sigma[d], 1e-6, 5.0)
                
                # Reset counters
                success_counts = np.zeros(dim)
                total_counts = np.zeros(dim)
                
                # Accept if offspring is better
                if best_offspring_f < f_current:
                    x_current = best_offspring
                    f_current = best_offspring_f
                    stagnation = 0
                else:
                    stagnation += 1
                
                # If stagnating too long, restart from global best with larger steps
                if stagnation > max_stagnation:
                    x_current = best_x.copy()
                    f_current = best_f
                    sigma = np.ones(dim) * 1.0
                    stagnation = 0

        # --- Phase 3: Final Polish ---
        # Use remaining budget for a final local search from the global best
        if calls < budget and time.time() < deadline:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_polish = np.ones(dim) * 0.1
            
            while calls < budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                x_candidate = x_polish + sigma_polish * z
                x_candidate = np.clip(x_candidate, -5, 5)
                
                f_candidate = counting_func(x_candidate)
                
                if f_candidate < f_polish:
                    x_polish = x_candidate
                    f_polish = f_candidate
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.9
                
                if sigma_polish < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
