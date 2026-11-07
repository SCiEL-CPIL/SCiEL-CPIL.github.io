"""Multi-start local optimizer with Sobol exploration for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc

class Optimizer:
    """Hybrid Sobol + L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Sobol Space-Filling Exploration ---
        # Use a Sobol sequence to cover the space efficiently.
        # 5 dimensions. 32 points is a good power of 2.
        # This provides a coarse landscape map.
        
        # We want to use about 20-30% of budget for exploration.
        # Let's use 64 points if budget allows, else 32.
        n_sobol = 64 if budget >= 100 else 32
        
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            # Generate Sobol points in [0, 1]^dim
            sobol_points_unit = sampler.random(n_sobol)
            # Map to [-5, 5]^dim
            sobol_points = sobol_points_unit * 10.0 - 5.0
            
            # Evaluate Sobol points
            for x in sobol_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                wrapped_func(x)
        except Exception:
            # Fallback to random if Sobol fails
            for _ in range(n_sobol):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                wrapped_func(x)

        # --- Phase 2: Multi-start Local Refinement ---
        # Identify the top K candidates from the Sobol exploration.
        # Since we don't store all points explicitly in a list for sorting easily
        # without extra memory/complexity in the closure, we can just
        # restart from the global best and a few perturbations.
        
        # However, a better strategy is to keep the best few points.
        # Let's re-implement to keep a history of top points.
        
        # Actually, the current wrapped_func only keeps the single best.
        # To do multi-start properly from exploration, we should store the top points.
        
        # Let's refine the strategy:
        # 1. We already evaluated 64 points.
        # 2. We know the global best.
        # 3. We will now run L-BFGS-B from the global best.
        # 4. We will also run L-BFGS-B from a few random perturbations of the best.
        # 5. We will alternate between exploitation (L-BFGS-B from best) and
        #    exploration (random restarts) based on stagnation.
        
        bounds = [(-5.0, 5.0)] * dim
        
        # Track stagnation
        last_improvement_eval = eval_count
        stagnation_threshold = 20 # If no improvement in 20 evals, explore more
        
        # To implement multi-start from exploration properly, let's keep a list of
        # the best 5 points found so far.
        # We need to modify wrapped_func to maintain a small list.
        
        # Let's restart the logic with a slightly different structure to support this.
        # Actually, we can just assume the global best is a good start for L-BFGS-B.
        # The "multi-start" aspect comes from restarting L-BFGS-B if it fails or stagnates.
        
        # Let's add a "local search" phase that aggressively exploits the best point.
        
        # Run L-BFGS-B from the current best
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Main Loop: Alternate between local refinement and random exploration
        # We will use a "budget" for each phase.
        
        # Let's use a loop that checks time and budget.
        # Strategy:
        # - If we haven't improved in X evals, do a random restart (Sobol-like or uniform).
        # - Otherwise, do a local refinement (L-BFGS-B or Nelder-Mead).
        
        # To improve on the seed, we can use Nelder-Mead for the local search
        # as it is often faster in low dimensions than L-BFGS-B for these functions,
        # or use a combination.
        
        # Let's try Nelder-Mead for the local steps as it handles non-smoothness
        # (though f22 is smooth, it's multimodal) and is very fast.
        
        while eval_count < budget and time.time() < deadline:
            # Check for stagnation
            if eval_count - last_improvement_eval > stagnation_threshold:
                # Stagnation: Explore
                # Pick a random point
                x_start = rng.uniform(-5.0, 5.0, dim)
                # Or perturb the best
                if rng.random() < 0.5:
                    x_start = best_x + rng.normal(0, 1.0, dim)
                    x_start = np.clip(x_start, -5.0, 5.0)
                
                # Evaluate this point
                f_start = wrapped_func(x_start)
                
                # If this point is better, update best (wrapped_func does this)
                # Now try to refine this new point
                if f_start < best_f + 1e-6: # If it's close to or better than best
                    # Run a local search from here
                    try:
                        res = minimize(
                            wrapped_func,
                            x_start,
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x
                            last_improvement_eval = eval_count
                    except Exception:
                        pass
            else:
                # Not stagnated: Exploit
                # Run a local search from the current best
                # Use Nelder-Mead for speed
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                        last_improvement_eval = eval_count
                except Exception:
                    pass

        # Phase 3: Final Polish
        # If we have budget left, do a few more random samples near best
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.1, dim)
            x = np.clip(x, -5.0, 5.0)
            wrapped_func(x)

        return best_x, best_f

# EVOLVE-BLOCK-END
