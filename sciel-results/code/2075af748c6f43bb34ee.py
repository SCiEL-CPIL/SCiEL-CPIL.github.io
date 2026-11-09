"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex optimizer with improved global search and consolidated local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Search
        # We evaluate a set of deterministic and random points to find a good starting basin.
        # For dim=5, we can afford ~50 evaluations for global search.
        # Points: Origin, +/- 4 in each dimension (structured), and random points.
        
        candidates = []
        # Origin
        candidates.append(np.zeros(dim))
        
        # Structured points: +/- 4 along each axis
        for d in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[d] = sign * 4.0
                candidates.append(x)
        
        # Random points
        rng = np.random.default_rng(42)
        n_random = 20
        for _ in range(n_random):
            candidates.append(rng.uniform(-5, 5, dim))
            
        best_x = None
        best_f = float('inf')
        
        for x in candidates:
            # Clip to box just in case, though candidates are within [-5,5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
                
        # Phase 2: Local Refinement
        # Use the remaining budget for a single, robust Nelder-Mead run.
        # Nelder-Mead with default settings in 5D takes roughly 5*dim + 1 = 26 evals for initial simplex.
        # Each iteration takes ~dim evals. 
        # With ~400 remaining budget, we can allow ~400/5 = 80 iterations.
        
        # We use 'Nelder-Mead' as it is robust for this class of problems if started well.
        # We set maxiter to a large number and rely on budget limits if we were implementing manually,
        # but scipy's minimize doesn't strictly limit function calls by a counter in options easily without a wrapper.
        # However, maxiter is a good proxy.
        
        # To be safe, we can wrap the function to count calls and stop if we exceed budget,
        # but the task says "More than 500 calls ... makes the whole result invalid".
        # So we must be careful.
        
        # Let's use a conservative maxiter for Nelder-Mead.
        # Initial simplex: dim+1 points.
        # Let's estimate calls: (dim+1) + maxiter * (dim/2) roughly.
        # 400 calls remaining.
        # Let's try maxiter=100.
        
        try:
            res = minimize(
                func, best_x, method='Nelder-Mead',
                options={
                    'maxiter': 100, 
                    'xatol': 1e-8, 
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res.fun < best_f:
                best_x = res.x.copy()
                best_f = res.fun
        except Exception:
            pass
        
        # Optional Phase 3: If we still have budget, do a very fine-tuned search
        # Check if we used all budget? Scipy doesn't report exact calls easily in all versions without wrapper.
        # But 100 iterations of NM in 5D is usually < 200 calls.
        # We have ~400 budget left after global search (50 calls).
        # So we have ~200 calls left.
        # Let's do another NM run with tighter tolerance?
        # Or just return.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
