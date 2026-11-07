"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Island (1+1)-ES with Error Repair for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        D = self.dim
        budget = self.budget

        # Track best
        best_x = np.zeros(D)
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-Island setup
        num_islands = 5
        islands = []
        
        # Initial positions: spread out in the box
        rng = np.random.RandomState(42)
        for i in range(num_islands):
            # Start with some spread
            x0 = rng.uniform(-3, 3, D)
            f0 = eval_func(x0)
            # Initial sigma: rough estimate based on distance to center
            dist = np.linalg.norm(x0)
            sigma0 = max(0.5, dist / D)
            islands.append({
                'x': x0,
                'f': f0,
                'sigma': sigma0,
                'last_improvement': 0
            })
        
        # Update global best
        if islands[0]['f'] < best_f:
            best_f = islands[0]['f']
            best_x = islands[0]['x'].copy()
        for isl in islands:
            if isl['f'] < best_f:
                best_f = isl['f']
                best_x = isl['x'].copy()

        # Parameters
        mu = 1 # (1+1)-ES
        # Error Repair: if no improvement, adjust sigma based on model
        # For Sphere, f(x) = ||x - x*||^2 + c
        # If we move by d, delta_f ~ 2 * <d, x - x*> + ||d||^2
        # We can estimate the gradient direction from recent failures or use a heuristic.
        # Here, we use a simple heuristic: if stuck, shrink sigma and try a random restart within a neighborhood.
        
        stagnation_limit = 20 # evaluations without improvement on an island
        repair_interval = 5 # check for repair every 5 evals per island

        try:
            while call_count < budget and time.time() < deadline:
                # Select island with lowest fitness (or round-robin with some randomization)
                # To maintain diversity, we can use a tournament or just cycle.
                # Let's cycle through islands, but prioritize those that have improved recently.
                # Simple approach: round-robin, but if an island is stuck, repair it.
                
                for i, isl in enumerate(islands):
                    if call_count >= budget or time.time() > deadline:
                        break
                    
                    # (1+1) step
                    y = rng.standard_normal(D)
                    new_x = isl['x'] + isl['sigma'] * y
                    new_x = np.clip(new_x, -5.0, 5.0)
                    new_f = eval_func(new_x)
                    
                    if new_f < isl['f']:
                        # Improvement
                        isl['x'] = new_x
                        isl['f'] = new_f
                        isl['last_improvement'] = call_count
                        # Expand sigma slightly on success
                        isl['sigma'] *= 1.1
                        # Update global best
                        if new_f < best_f:
                            best_f = new_f
                            best_x = new_x.copy()
                    else:
                        # No improvement
                        # Shrink sigma
                        isl['sigma'] *= 0.9
                        # Check for stagnation and apply Error Repair
                        if call_count - isl['last_improvement'] > stagnation_limit and call_count % repair_interval == 0:
                            # Error Repair: 
                            # 1. Reset sigma to a moderate value
                            # 2. Move to a random point in a small neighborhood of the current best on this island
                            # 3. Or, use a model-based estimate: 
                            #    If we assume Sphere, and we have two points, we can estimate the direction.
                            #    For simplicity, we'll do a local restart: jump to a random point within a radius of 1.0
                            #    This is a structural change: "Error Repair" via local restart with model-informed radius.
                            # The radius is informed by the current error magnitude (approximated by f - min_possible).
                            # Since we don't know f_opt, we use a heuristic: if f is large, restart wider.
                            radius = min(2.0, max(0.5, np.sqrt(isl['f']) / 10.0))
                            new_x = isl['x'] + rng.uniform(-radius, radius, D)
                            new_x = np.clip(new_x, -5.0, 5.0)
                            new_f = eval_func(new_x)
                            if new_f < isl['f']:
                                isl['x'] = new_x
                                isl['f'] = new_f
                                isl['last_improvement'] = call_count
                                if new_f < best_f:
                                    best_f = new_f
                                    best_x = new_x.copy()
                            else:
                                # If repair fails, shrink sigma more aggressively
                                isl['sigma'] *= 0.5
                            
                            # Ensure sigma doesn't get too small
                            isl['sigma'] = max(isl['sigma'], 1e-4)

        except (TimeoutError, ValueError):
            pass
        except Exception:
            pass

        return (best_x, best_f)
# EVOLVE-BLOCK-END
