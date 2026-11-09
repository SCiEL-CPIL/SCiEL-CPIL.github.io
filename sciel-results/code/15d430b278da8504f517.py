# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer combining deterministic multi-start local search with
    adaptive (1+1)-ES for the BBOB f13 Sharp Ridge function.
    
    Strategy:
    1. Initial phase: Use adaptive (1+1)-ES from center to get close to minimum.
    2. Transition: When ES stalls or budget allows, switch to deterministic 
       coordinate-wise line search (golden-section) for precise convergence.
    3. Final polish: Use remaining budget for fine-grained coordinate search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Adaptive (1+1)-ES with stall detection
        window_size = 5
        successes = 0
        steps_in_window = 0
        stall_threshold = 30  # Lower threshold to trigger transition earlier
        max_sigma = 5.0
        min_sigma = 1e-4
        
        # Initial state
        x = np.zeros(n)
        fx = wrapped_func(x)
        sigma = 1.5
        
        steps_without_improvement = 0
        es_phase = True
        
        # Phase 1: Run (1+1)-ES until stall or time/budget constraint
        while calls_made[0] < budget and es_phase:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = wrapped_func(y)
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                steps_without_improvement = 0
                sigma = min(sigma * 1.15, max_sigma)
            else:
                steps_without_improvement += 1
                sigma = max(sigma * 0.88, min_sigma)

            steps_in_window += 1
            
            # Adapt step size based on 1/5 rule
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 1/5:
                    sigma = min(sigma * 1.05, max_sigma)
                else:
                    sigma = max(sigma * 0.95, min_sigma)
                successes = 0
                steps_in_window = 0

            # Stall detection: transition to Phase 2
            if steps_without_improvement > stall_threshold:
                es_phase = False
                # Use the best point found so far for Phase 2
                x = best_x.copy()
                fx = best_f

        # Phase 2: Deterministic coordinate-wise line search (Powell-like)
        # This exploits the axis-aligned nature of the sharp ridge
        # Use golden-section search along each coordinate sequentially
        
        # Define golden-section search along one coordinate
        def golden_section_1d(x, coord_idx, lo, hi, func, max_iter=20):
            """
            Perform golden-section search along coordinate 'coord_idx' 
            between 'lo' and 'hi' for function 'func'.
            Returns the best x found and its function value.
            """
            if calls_made[0] >= budget:
                return x, func(x)
            
            # Golden ratio
            gr = (np.sqrt(5) - 1) / 2  # ~0.618
            a, b = lo, hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Evaluate initial points
            x_c = x.copy()
            x_d = x.copy()
            x_c[coord_idx] = c
            x_d[coord_idx] = d
            
            fc = wrapped_func(x_c)
            fd = wrapped_func(x_d)
            
            best_local_x = x_c.copy() if fc < fd else x_d.copy()
            best_local_f = min(fc, fd)
            
            for _ in range(max_iter):
                if calls_made[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    x_c = x.copy()
                    x_c[coord_idx] = c
                    fc = wrapped_func(x_c)
                    if fc < best_local_f:
                        best_local_f = fc
                        best_local_x = x_c.copy()
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    x_d = x.copy()
                    x_d[coord_idx] = d
                    fd = wrapped_func(x_d)
                    if fd < best_local_f:
                        best_local_f = fd
                        best_local_x = x_d.copy()
                
                # Check convergence
                if abs(b - a) < 1e-6:
                    break
            
            return best_local_x, best_local_f

        # Perform sequential coordinate search
        # Multiple passes to handle interactions between coordinates
        current_x = x.copy()
        current_f = fx
        
        # Determine step size for line search bounds
        # Use a reasonable range around current point
        search_radius = max(0.5, sigma * 2) if sigma > 0.1 else 1.0
        
        # Multiple passes of coordinate-wise optimization
        num_passes = 3  # Can adjust based on remaining budget
        
        for pass_num in range(num_passes):
            if calls_made[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            for coord in range(n):
                if calls_made[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Define search bounds for this coordinate
                lo = max(-5.0, current_x[coord] - search_radius)
                hi = min(5.0, current_x[coord] + search_radius)
                
                if hi - lo < 1e-8:
                    continue
                
                # Perform golden-section search along this coordinate
                new_x, new_f = golden_section_1d(
                    current_x, coord, lo, hi, func, 
                    max_iter=15
                )
                
                if new_f < current_f:
                    current_x = new_x
                    current_f = new_f
                
                # Shrink search radius for next pass
                search_radius *= 0.5
        
        # Phase 3: Final fine-tuning with small (1+1)-ES steps
        # Use remaining budget for very fine-grained search
        if calls_made[0] < budget and time.time() < deadline:
            x = current_x
            fx = current_f
            sigma = 0.05  # Very small steps for final polish
            steps_without_improvement = 0
            fine_stall_threshold = 15
            
            while calls_made[0] < budget and steps_without_improvement < fine_stall_threshold:
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    steps_without_improvement = 0
                    sigma = min(sigma * 1.1, 0.1)
                else:
                    steps_without_improvement += 1
                    sigma = max(sigma * 0.9, 1e-5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
