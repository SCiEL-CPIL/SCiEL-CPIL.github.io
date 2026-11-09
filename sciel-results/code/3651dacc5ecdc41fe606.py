# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Coordinate-wise Line Search (Powell-like) Optimizer.
    
    Strategy:
    1. Initialize with a small set of random points to find a good starting basin.
    2. Perform a sequence of 1D line searches along the coordinate axes.
       This is a "derivative-free" approach that effectively adapts to the
       local curvature by finding the optimal step along each axis.
       For "Sharp Ridge" functions, this is often more efficient than isotropic
       Gaussian steps (like in ES) because it can take large steps along flat
       directions and small steps along sharp directions without wasting
       evaluations on failed large steps in all directions simultaneously.
    3. Repeat the sequence of axis searches until budget is exhausted.
    4. Include a simple "stall detection" that triggers a random restart
       or a small perturbation if progress stalls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Use 20 random points to find a good starting point.
        n_init = 20
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            timed_call(x)
            
        if best_x is None:
            # Fallback
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made += 1
            return x, fx

        x = best_x.copy()
        fx = best_f

        # Phase 2: Coordinate-wise Line Search
        # We will iterate through the axes. For each axis, we perform a golden-section search
        # or a simple bracketing search to find the minimum along that axis.
        # To keep it simple and robust, we will use a "step-and-halve" or "golden section" 
        # approach. Golden section is efficient (uses ~1.6 calls per factor of 0.618 reduction).
        
        # We'll use a simple heuristic: 
        # Try a step in + and - direction. If it improves, expand. If not, shrink.
        # This is a form of "hill climbing" with adaptive step size per axis.
        
        # Let's maintain a step size for each axis.
        steps = np.ones(self.dim) * 1.0
        
        # We will loop until budget runs out.
        # In each "cycle", we visit each axis once.
        
        no_improvement_count = 0
        max_no_improvement = 50 # If we fail to improve for 50 axis visits, try a random jump
        
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
            
            improved_in_cycle = False
            
            for d in range(self.dim):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline - 2:
                    break
                
                # Perform a 1D search along axis d
                # We will try to find a better point along this axis using a few evaluations.
                # Simple strategy: try x + step, x - step.
                # If one is better, move there and expand step.
                # If neither is better, shrink step.
                
                x_plus = x.copy()
                x_plus[d] += steps[d]
                if x_plus[d] > 5: x_plus[d] = 5
                if x_plus[d] < -5: x_plus[d] = -5
                
                x_minus = x.copy()
                x_minus[d] -= steps[d]
                if x_minus[d] > 5: x_minus[d] = 5
                if x_minus[d] < -5: x_minus[d] = -5
                
                # Evaluate both (if budget allows)
                # To save calls, we might just pick one direction first, but evaluating both
                # gives us a better idea of the gradient direction.
                # Let's evaluate both.
                
                fx_plus = timed_call(x_plus)
                if fx_plus is None:
                    break
                
                fx_minus = timed_call(x_minus)
                if fx_minus is None:
                    break
                
                # Compare
                if fx_plus < fx and fx_plus < fx_minus:
                    x = x_plus
                    fx = fx_plus
                    steps[d] *= 1.5 # Expand
                    improved_in_cycle = True
                elif fx_minus < fx and fx_minus < fx_plus:
                    x = x_minus
                    fx = fx_minus
                    steps[d] *= 1.5 # Expand
                    improved_in_cycle = True
                else:
                    # Neither improved
                    steps[d] *= 0.5 # Shrink
                    if steps[d] < 1e-4:
                        steps[d] = 1e-4 # Prevent underflow

            if improved_in_cycle:
                no_improvement_count = 0
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    # Stagnation: try a random perturbation
                    # Pick a random axis and make a random jump
                    d = rng.integers(self.dim)
                    jump = rng.uniform(-2, 2)
                    x[d] += jump
                    if x[d] > 5: x[d] = 5
                    if x[d] < -5: x[d] = -5
                    fx = timed_call(x)
                    if fx is None:
                        break
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Reset steps for the jumped axis
                    steps[d] = 1.0
                    no_improvement_count = 0

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
