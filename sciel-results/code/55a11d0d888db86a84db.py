"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead simplex with adaptive reset and multi-start diversification."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        start_time = time.time()
        max_time = 285.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        
        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            calls_used += 1
            return func(x)

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, budget_remaining, sigma_init=0.5):
            nonlocal calls_used, global_best_x, global_best_f
            
            # Initialize simplex
            # Vertices: x0 + sigma * basis vectors
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(dim):
                v = x0.copy()
                v[i] += sigma_init
                simplex.append(v)
                fval = safe_call(v)
                fvals.append(fval)
                
                if fval < global_best_f:
                    global_best_f = fval
                    global_best_x = v.copy()

            # Coefficients
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Main loop
            while calls_used < total_budget and (time.time() - start_time) < max_time:
                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check for convergence (optional, but good for early exit if stuck)
                # If the range of function values is very small, we might be stuck
                # But for now, let's just run until budget or time.
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)
                
                if fr < fvals[-1]:
                    if fr < fvals[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = safe_call(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        # Accept reflection
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if reflection was bad, inside otherwise
                    # Standard NM: if reflection is worse than best, do inside contraction
                    # If reflection is worse than worst, do outside contraction?
                    # Let's use standard: 
                    # If fr > fvals[-1] (worst), we need contraction.
                    # If fr < fvals[0] (best), we did expansion.
                    # Here fr >= fvals[-1].
                    
                    # Check if it's outside or inside contraction
                    # Usually: if fr > fvals[-1] (which it is), check against fvals[0]
                    # If fr < fvals[0] is false, we are in the "middle" or "bad" case.
                    # Standard NM: 
                    # 1. If fr < fvals[0]: Expand
                    # 2. Else if fr < fvals[-1]: Accept Reflection
                    # 3. Else: Contract
                    #    - If fr < fvals[-1]: Inside contraction (xr is better than worst)
                    #    - Else: Outside contraction (xr is worse than worst)
                    
                    if fr < fvals[-1]:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        fc = safe_call(xc)
                        if fc < fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = safe_call(simplex[i])
                    else:
                        # Outside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                        fc = safe_call(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = safe_call(simplex[i])
                
                # Update global best
                for i in range(len(fvals)):
                    if fvals[i] < global_best_f:
                        global_best_f = fvals[i]
                        global_best_x = simplex[i].copy()

        # Multi-start strategy
        # We divide the budget into several restarts.
        # Start 1: Center
        # Start 2-5: Random points
        
        num_starts = 5
        # Allocate budget roughly equally, but give more to later starts if early ones are bad?
        # Or just equal splits.
        # Let's try equal splits first.
        budget_per_start = total_budget // num_starts
        
        # First start at center
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if f0 < global_best_f:
            global_best_f = f0
            global_best_x = x0.copy()
            
        nelder_mead(x0, f0, budget_per_start)
        
        # Subsequent starts at random points
        for _ in range(num_starts - 1):
            if time.time() - start_time > max_time:
                break
            if calls_used >= total_budget:
                break
                
            x0 = rng.uniform(-5, 5, size=dim)
            f0 = safe_call(x0)
            if f0 < global_best_f:
                global_best_f = f0
                global_best_x = x0.copy()
                
            # Use remaining budget for this start
            nelder_mead(x0, f0, total_budget)

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
