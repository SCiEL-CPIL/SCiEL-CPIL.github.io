import numpy as np
import time

class Optimizer:
    """
    Deterministic Coarse-to-Fine Halton Grid Search.
    
    This strategy abandons stochastic local search in favor of a systematic
    space-filling sampling approach using the Halton sequence.
    1. It generates a Halton sequence over the search space [-5, 5]^dim.
    2. It evaluates points sequentially until the budget is exhausted or a 
       time limit is reached.
    3. Crucially, it implements a "refinement" logic: if the search has 
       progressed significantly (e.g., used 50% of budget) and time remains,
       it restarts the Halton sequence but mapped to a smaller box centered 
       on the current best point. This effectively performs a coarse-to-fine 
       search without the stochastic traps of random restarts.
    
    This is structurally different from Basin Hopping because:
    1. It uses deterministic low-discrepancy sequences (Halton) instead of 
       random normal perturbations.
    2. It does not maintain a population of independent basins; it maintains 
       a single global best and refines the search region around it.
    3. It guarantees uniform coverage of the initial domain, which helps 
       avoid missing global minima in multimodal landscapes like Rastrigin 
       or Gallagher, provided the initial grid resolution is sufficient.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, count, dim, base_offset=0):
        """
        Generate a Halton sequence of 'count' points in 'dim' dimensions.
        The points are in [0, 1)^dim.
        """
        # Primes for the first few dimensions
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # If dim > 10, we would need more primes, but dim is fixed to 5 in the task.
        # We handle dim up to 10 for safety, but task says dim=5.
        
        points = np.zeros((count, dim))
        
        for d in range(dim):
            prime = primes[d]
            result = np.zeros(count)
            factor = 1.0 / prime
            for i in range(1, count + 1):
                n = i + base_offset
                while n % prime == 0:
                    n //= prime
                    factor /= prime
                result[i - 1] = n * factor
            points[:, d] = result
            
        return points

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 325s out of 330s limit
        deadline = start_time + 325.0
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        
        # Initial search bounds
        lower_bound = np.full(dim, -5.0)
        upper_bound = np.full(dim, 5.0)
        
        global_best_x = None
        global_best_fx = np.inf
        
        # We will use a coarse-to-fine approach.
        # Phase 1: Broad exploration.
        # We split the budget into chunks.
        # If we find a good point, we refine the search space around it.
        
        # Define a list of "phases" with fractions of the budget and shrink factors.
        # Phase 1: 40% budget, full space.
        # Phase 2: 30% budget, 50% size space around best.
        # Phase 3: 20% budget, 25% size space around best.
        # Phase 4: 10% budget, 10% size space around best.
        
        # However, we don't know if we'll find a good point early.
        # So we iterate through phases, but only shrink if we have a best point.
        
        phases = [
            (0.4, 1.0),   # 40% budget, 100% size
            (0.3, 0.5),   # 30% budget, 50% size
            (0.2, 0.25),  # 20% budget, 25% size
            (0.1, 0.10)   # 10% budget, 10% size
        ]
        
        # To make the Halton sequence restartable and consistent, we keep track of 
        # how many points we've generated in the current phase's coordinate system.
        # But simpler: just generate a new Halton sequence for each phase, 
        # mapped to the current box.
        
        # We need to be careful not to exceed the total budget.
        
        for frac_budget, shrink_factor in phases:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Determine how many points to evaluate in this phase
            # We reserve a small buffer for time safety.
            # Let's just use the remaining budget if it's less than the phase allocation,
            # otherwise use the phase allocation.
            
            allocated = int(budget * frac_budget)
            # Don't exceed remaining budget
            remaining = budget - calls_used
            num_points = min(allocated, remaining)
            
            if num_points <= 0:
                break
            
            # Define the current search box
            if global_best_x is None:
                # First phase, use full box
                center = np.zeros(dim)
                size = 10.0 # Full range
            else:
                center = global_best_x
                # Shrink the box around the best point
                # Original size was 10.0.
                # We want the box to be centered at best, with side length = 10.0 * shrink_factor
                # But we must keep it within [-5, 5]^dim
                size = 10.0 * shrink_factor
            
            half_size = size / 2.0
            lower = center - half_size
            upper = center + half_size
            
            # Clip to valid domain
            lower = np.clip(lower, -5.0, 5.0)
            upper = np.clip(upper, -5.0, 5.0)
            
            # Ensure lower <= upper
            # If center is near boundary, clipping might make lower > upper? No, clip is element-wise.
            # But if center is 4.9 and half_size is 5, lower is -0.1, upper is 9.9 -> clip to 5.
            # lower = -0.1, upper = 5.0. Valid.
            
            # Generate Halton points in [0, 1]^dim
            halton_pts = self._halton_sequence(num_points, dim)
            
            # Map to [lower, upper]
            # point = lower + halton_pt * (upper - lower)
            span = upper - lower
            mapped_pts = lower + halton_pts * span
            
            # Evaluate points
            for i in range(num_points):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x = mapped_pts[i]
                # Ensure strict bounds just in case of floating point issues
                x = np.clip(x, -5.0, 5.0)
                
                fx = func(x)
                calls_used += 1
                
                if fx < global_best_fx:
                    global_best_fx = fx
                    global_best_x = x.copy()
        
        # If we still have budget and time, do a final polish?
        # The phases should cover the budget.
        # But if we broke early due to time, we return what we have.
        
        return global_best_x, global_best_fx
