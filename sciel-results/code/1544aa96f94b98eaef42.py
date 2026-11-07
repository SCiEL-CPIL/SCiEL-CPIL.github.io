"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Deterministic Sobol Global Search + Local (1+1)-ES Refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility in the deterministic phase, 
        # but the Sobol generator is deterministic by nature.
        # We don't really need a RNG for Sobol, but we might for tie-breaking or minor noise if needed.
        # For this strategy, we rely on the structure.
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            # Clip to ensure we are in domain, though func does it too
            x_eval = np.clip(x, -5, 5)
            fx = func(x_eval)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_eval.copy()
            return fx

        # Phase 1: Global Exploration with Sobol Sequence
        # Allocate ~60% of the budget to global search.
        # Sobol sequences are space-filling and deterministic.
        # For dim=5, we can generate a sufficient number of points.
        n_global = int(budget * 0.6)
        
        # Generate Sobol sequence
        # qmc.Sobol requires dim <= 21 for standard implementation? 
        # Actually scipy.stats.qmc.Sobol supports up to 21 dimensions? 
        # Let's check. Yes, typically 21. dim=5 is fine.
        sobol = qmc.Sobol(d=dim, scramble=False)
        # We want n_global points.
        # Sobol generates points in [0, 1]^d.
        # We need to map to [-5, 5]^d.
        
        # Generate points. 
        # Note: Sobol sequence includes the origin (0,...,0) as the first point if not shifted?
        # Actually, standard Sobol starts at 0. 
        # We can generate n_global points.
        sobol_points = sobol.random(n_global)
        
        # Map [0, 1] to [-5, 5]
        # x = -5 + 10 * sobol_point
        sobol_points_mapped = -5 + 10 * sobol_points
        
        # Evaluate all global points
        for i in range(n_global):
            if time_left() < 1.0:
                break
            x = sobol_points_mapped[i]
            evaluate(x)
            
        # Phase 2: Local Refinement
        # Use the remaining budget for local search starting from the best point found.
        # We can also try a few other good points from the global phase if budget allows,
        # but focusing on the single best is usually more effective for continuous optimization 
        # once the basin is identified.
        
        if best_x is not None and time_left() > 5.0 and calls_made < budget:
            # Local (1+1)-ES with 1/5 rule
            x = best_x.copy()
            f = best_f
            sigma = 1.0  # Initial step size
            
            # Adaptation parameters
            # c_sigma = 1/5
            # We use a window for success rate
            window_size = 20
            success_history = np.zeros(window_size, dtype=bool)
            success_idx = 0
            gen_count = 0
            successes_in_window = 0
            
            # Stagnation detection
            stagnation_limit = 30
            stagnation_count = 0
            
            while calls_made < budget and time_left() > 1.0:
                # Generate trial point
                # Use a fixed RNG for local search to ensure determinism? 
                # Or just use np.random with a fixed seed?
                # The task says "deterministic" for the strategy, but the local search 
                # usually involves random directions. 
                # To be strictly deterministic, we could use a fixed sequence of directions 
                # or a seeded RNG. Let's use a seeded RNG for the local phase.
                # However, creating a new RNG each time __call__ is invoked with a fixed seed 
                # makes the local phase deterministic for a given global best.
                local_rng = np.random.default_rng(42)
                
                z = local_rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = func(x_trial)
                calls_made += 1
                
                accepted = False
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    accepted = True
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    accepted = False
                    stagnation_count += 1
                
                # Update 1/5 rule
                if gen_count >= window_size:
                    if success_history[success_idx]:
                        successes_in_window -= 1
                
                success_history[success_idx] = accepted
                if accepted:
                    successes_in_window += 1
                    
                gen_count += 1
                success_idx = (success_idx + 1) % window_size
                
                # Update sigma
                n_eff = min(gen_count, window_size)
                success_rate = successes_in_window / n_eff
                
                c_sigma = 0.2
                sigma = sigma * np.exp(c_sigma * (success_rate - 0.2))
                
                # Bound sigma
                sigma = np.clip(sigma, 1e-6, 5.0)
                
                if stagnation_count > stagnation_limit:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
