"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Hybrid optimizer for Bent Cigar (BBOB f12):
    1. Sobol global sampling to locate the basin of attraction.
    2. Adaptive Coordinate Descent (Powell-style) with per-dimension step sizes
       to efficiently navigate the anisotropic narrow valley.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_used = 0
        best_x = np.zeros(dim)
        best_fx = np.inf
        
        def evaluate(x):
            nonlocal calls_used, best_x, best_fx
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            calls_used += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use low-discrepancy sampling to ensure good coverage of [-5, 5]^dim
        # 64 points is a good balance between coverage and budget consumption
        n_global = 64
        try:
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            samples = sobol.random(n_global)
        except Exception:
            # Fallback to random if scipy qmc fails
            samples = rng.uniform(size=(n_global, dim))
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        for x in samples:
            if time.time() > deadline or calls_used >= budget:
                break
            fx = evaluate(x)
            if np.isinf(fx):
                break
        
        # If no valid points were found (shouldn't happen), start at center
        if best_x is None or np.all(best_x == 0) and best_fx == np.inf:
            x0 = np.zeros(dim)
            evaluate(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Adaptive Coordinate Descent ---
        # Powell's method is particularly effective for anisotropic functions like Bent Cigar
        # because it adapts step sizes per dimension, allowing rapid convergence along the
        # narrow valley while maintaining stability in wider directions.
        
        # Initialize step sizes. We start with a moderate step.
        # The Sobol phase ensures we are likely in the basin, so we can start relatively small
        # but not too small to allow for exploration if we missed the exact valley center.
        step_sizes = np.full(dim, 0.5)
        
        # Current point
        x = x0.copy()
        fx = evaluate(x) if not (x0 is best_x) else best_fx # Re-evaluate if needed, but we track best_x
        
        # Actually, let's just re-evaluate the current best to be sure, or use the tracked value.
        # Since evaluate() updates best_x/best_fx, and x0 is best_x, fx is best_fx.
        
        # Powell's method:
        # 1. Minimize along each coordinate direction
        # 2. Check for improvement from the previous cycle
        # 3. If improved, move along the direction of greatest improvement and add it to the set
        # 4. Remove the worst direction
        
        # We will implement a simplified adaptive coordinate descent that is robust to bounds.
        # Standard Powell can be tricky with bounds. A simpler "Adaptive Step Size" per coordinate
        # often works well for Bent Cigar.
        
        # Let's use a robust adaptive 1D search per coordinate.
        # For each coordinate i:
        #   - Try x[i] + step[i], x[i] - step[i]
        #   - If improvement, adjust step[i] (increase if improved significantly, decrease if not)
        #   - Use a simple golden section or bisection if needed? 
        #     For Bent Cigar, simple adaptive steps usually suffice because the function is unimodal in each coordinate
        #     given the others are fixed (roughly).
        
        # Let's implement a simple adaptive coordinate descent.
        # This is faster and more robust than full Powell with bounds.
        
        # Initialize steps based on the spread of the Sobol samples or just a heuristic.
        # Let's start with 0.5 for all.
        
        # To make it more "Powell-like" or effective, we can track the history of improvements.
        # But simple adaptive steps are very strong on BBOB f12.
        
        # We will alternate between:
        # 1. Coordinate descent with adaptive steps.
        # 2. If stuck, try a small random perturbation (jitter) to escape local minima (though Bent Cigar has only one).
        
        # Let's stick to pure adaptive coordinate descent for speed and reliability.
        
        # State for adaptive steps
        # We want to quickly find the right scale for each dimension.
        # Start with a large step, shrink if no improvement, expand if improvement is large.
        
        # Re-initialize steps to be larger for the first few iterations to ensure we cover the valley
        # if the Sobol start was slightly off.
        step_sizes = np.full(dim, 1.0)
        
        x = best_x.copy()
        fx = best_fx
        
        # Main loop for refinement
        # We have budget - calls_used left.
        # Each cycle uses at most dim calls (one per coordinate).
        # So we can do roughly (budget - calls_used) / dim cycles.
        
        max_cycles = 10000 # Safety cap
        cycle = 0
        
        while calls_used < budget and time.time() < deadline and cycle < max_cycles:
            cycle += 1
            improved_cycle = False
            min_fx_this_cycle = fx
            
            for i in range(dim):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                x_try = x.copy()
                x_try[i] += step_sizes[i]
                x_try = np.clip(x_try, -5, 5)
                
                fx_try = evaluate(x_try)
                
                if fx_try < fx:
                    x = x_try
                    fx = fx_try
                    improved_cycle = True
                    # If improvement is significant, we might want to increase step?
                    # Or just keep it.
                    # Simple heuristic: if we improved, keep step or slightly increase?
                    # If we didn't improve, shrink step.
                    # Let's shrink step if we didn't improve in this direction.
                    # If we did improve, we don't know if we overshot or not.
                    # For Bent Cigar, shrinking is safer.
                    
                    # Let's try a negative step as well to see if we can improve further?
                    # No, that doubles the cost. 
                    # Better: if fx_try < fx, we moved. 
                    # We can try to see if a larger step was better? No, too expensive.
                    # We can try a smaller step to fine tune? 
                    # Actually, standard adaptive ES logic:
                    # If success, increase sigma. If fail, decrease sigma.
                    # Here, per coordinate:
                    # If fx_try < fx: success. step_sizes[i] *= 1.2 ?
                    # Else: fail. step_sizes[i] *= 0.5 ?
                    
                    # Let's use a more aggressive adaptation.
                    # If we improved, we might have undershot. Try a larger step?
                    # Or just accept and shrink for next time to converge?
                    # Convergence is key for AOCC.
                    
                    # Let's try: if improved, keep step. If not, shrink.
                    # But this is slow to converge if step is too large initially.
                    # Let's do:
                    # 1. Try +step
                    # 2. If improved, accept. Then try +step/2 to see if we can improve more? 
                    #    No, that's a line search.
                    
                    # Let's use a simple 1D golden section search if we detect improvement?
                    # That's complex.
                    
                    # Let's stick to:
                    # If improved: step_sizes[i] = step_sizes[i] * 1.1 (slight increase to explore)
                    # If not improved: step_sizes[i] = step_sizes[i] * 0.5 (shrink)
                    
                    step_sizes[i] *= 1.1
                    if step_sizes[i] > 5.0: step_sizes[i] = 5.0
                    
                else:
                    # Try negative step
                    x_try = x.copy()
                    x_try[i] -= step_sizes[i]
                    x_try = np.clip(x_try, -5, 5)
                    
                    fx_try = evaluate(x_try)
                    
                    if fx_try < fx:
                        x = x_try
                        fx = fx_try
                        improved_cycle = True
                        step_sizes[i] *= 1.1
                        if step_sizes[i] > 5.0: step_sizes[i] = 5.0
                    else:
                        # No improvement in either direction
                        step_sizes[i] *= 0.5
                        if step_sizes[i] < 1e-8:
                            step_sizes[i] = 1e-8
                        
            if not improved_cycle:
                # If no coordinate improved, we are likely at a local minimum (or global)
                # or steps are too small.
                # If steps are very small, break.
                if np.all(step_sizes < 1e-6):
                    break
                # Otherwise, try a random perturbation to escape potential flat spots?
                # Bent Cigar is smooth, so this shouldn't happen unless we are very close.
                # Let's just break if no improvement and steps are small.
                if np.all(step_sizes < 1e-4):
                    break
                # If steps are still large but no improvement, something is wrong?
                # Maybe we are at a boundary?
                # Let's shrink all steps aggressively.
                step_sizes *= 0.5
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
