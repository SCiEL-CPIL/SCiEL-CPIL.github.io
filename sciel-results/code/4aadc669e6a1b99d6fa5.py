"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES and Coordinate Descent for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # Phase 1: (1+1)-ES to find a good basin
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        budget = self.budget
        dim = self.dim
        
        sigma = 2.0
        es_budget = 200  # Use 200 calls for ES, leaving 300 for CD
        
        for _ in range(min(es_budget, budget - 1)):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2  # Increase step size slightly on success
            else:
                sigma *= 0.85  # Decrease step size on failure
            
            if calls >= budget:
                break

        # Phase 2: Coordinate Descent with Golden Section Search
        # Use remaining budget
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return x, fx

        # Golden Section Search parameters
        # We have roughly 300 calls left.
        # D=5. 
        # Let's do passes until we run out of budget or time.
        # In each pass, we optimize each dimension.
        # Cost per dimension: ~20-30 calls.
        # Let's allocate calls per dimension dynamically or fixed.
        # Fixed: 20 calls per dimension per pass.
        # Pass cost: 5 * 20 = 100 calls.
        # We can do ~3 passes with 300 calls.
        
        gs_iters = 25
        calls_per_dim = gs_iters + 2 # 2 initial evals + (iters-2) updates? No, GS loop logic.
        # Standard GS:
        # 2 initial evals.
        # Each subsequent step adds 1 eval.
        # Total evals = 2 + (gs_iters - 2) = gs_iters? 
        # Let's say we want gs_iters evaluations per dimension.
        
        r = (np.sqrt(5.0) - 1.0) / 2.0
        
        # Determine number of passes
        calls_per_pass = dim * gs_iters
        max_passes = max(1, remaining_budget // calls_per_pass)
        
        # If we have leftover calls, we can do a few more dimensions or increase precision
        # But for simplicity, stick to passes.
        
        for _ in range(max_passes):
            if time.time() > deadline or calls >= budget:
                break
            
            for i in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Optimize dimension i
                # Search range: [-5, 5]
                a = -5.0
                b = 5.0
                
                # We can tighten the bracket if we know the current x[i] is good.
                # But Bent Cigar is convex, so global min is in [-5,5].
                # However, local min might be far. 
                # Actually, since we just did ES, x is likely close.
                # We can restrict search to [x[i]-1, x[i]+1] clipped to [-5,5]?
                # Risky if ES failed to get close.
                # Let's stick to full range for safety in first pass, 
                # then narrow in subsequent passes?
                # For now, full range is safer.
                
                best_x_i = x[i]
                best_f_i = fx
                
                c = a + (1.0 - r) * (b - a)
                d = a + r * (b - a)
                
                x_c = x.copy()
                x_c[i] = c
                if calls < budget and time.time() < deadline:
                    f_c = func(x_c)
                    calls += 1
                else:
                    f_c = np.inf
                
                x_d = x.copy()
                x_d[i] = d
                if calls < budget and time.time() < deadline:
                    f_d = func(x_d)
                    calls += 1
                else:
                    f_d = np.inf
                    
                if f_c < best_f_i:
                    best_f_i = f_c
                    best_x_i = c
                if f_d < best_f_i:
                    best_f_i = f_d
                    best_x_i = d
                    
                # Perform GS iterations
                for _ in range(gs_iters - 2):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    if f_c < f_d:
                        b = d
                        d = c
                        f_d = f_c
                        c = a + (1.0 - r) * (b - a)
                        x_c = x.copy()
                        x_c[i] = c
                        f_c = func(x_c)
                        calls += 1
                        if f_c < best_f_i:
                            best_f_i = f_c
                            best_x_i = c
                    else:
                        a = c
                        c = d
                        f_c = f_d
                        d = a + r * (b - a)
                        x_d = x.copy()
                        x_d[i] = d
                        f_d = func(x_d)
                        calls += 1
                        if f_d < best_f_i:
                            best_f_i = f_d
                            best_x_i = d

                # Update x and fx with the best found in this dimension
                x[i] = best_x_i
                fx = best_f_i
                
        return x, fx
# EVOLVE-BLOCK-END
