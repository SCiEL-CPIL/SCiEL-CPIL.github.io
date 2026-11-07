"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        rng = np.random.default_rng(42)
        
        # Initial point: random in the box
        x0 = rng.uniform(-5, 5, self.dim)
        
        # Initialize simplex
        # The simplex has dim+1 vertices
        # We start with a small perturbation to ensure non-degeneracy
        # Scale of initial simplex: 0.5 is reasonable for [-5, 5]
        alpha = 0.5
        beta = 0.5
        
        # Vertices
        # v0 = x0
        # vi = x0 + alpha * e_i for i=1..dim
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        for i in range(1, self.dim + 1):
            simplex[i] = x0
            simplex[i, i-1] += alpha
            
        # Evaluate initial simplex
        fvals = np.array([func(v) for v in simplex])
        
        # Track best
        best_x = simplex[np.argmin(fvals)]
        best_f = np.min(fvals)
        
        # Standard NM parameters
        r = 1.0   # Reflection
        e = 2.0   # Expansion
        c = 0.5   # Contraction
        s = 0.5   # Shrinkage
        
        calls_made = self.dim + 1
        budget_left = self.budget - calls_made
        
        while budget_left > 0:
            # Check time
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
            
            # Check convergence: if the range of fvals is very small, we might be done
            # But we must use the budget effectively. 
            # A common heuristic: if max_f - min_f < 1e-10, stop.
            # However, for scoring, early progress counts. We should keep going until budget or time.
            
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + r * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made += 1
            budget_left -= 1
            
            if fr < fvals[0]:
                # Try expansion
                xe = centroid + e * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made += 1
                budget_left -= 1
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                # Outside contraction if fr < fvals[-1], else inside
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + c * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    budget_left -= 1
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + s * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            budget_left -= 1
                            if budget_left <= 0:
                                break
                        if budget_left <= 0:
                            break
                else:
                    # Inside contraction
                    xc = centroid - c * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    budget_left -= 1
                    
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + s * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            budget_left -= 1
                            if budget_left <= 0:
                                break
                        if budget_left <= 0:
                            break
            
            # Optional: If simplex is too small, we might get stuck due to numerical precision.
            # But with 500 budget, we are unlikely to hit precision limits unless we converge very fast.
            # If we converge very fast, we should probably just return the best found.
            # Let's add a simple check: if the spread of coordinates is < 1e-8, we can stop early 
            # to save time, or just let it run out of budget. 
            # Given the scoring rewards early progress, running out of budget is fine as long as we found the min.
            # However, if we are stuck in a flat area, NM might oscillate. 
            # A restart strategy could be added, but let's keep it simple first.
            
            # If budget is exhausted, exit loop
            if budget_left <= 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
