"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using multi-start Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # 1. Generate a diverse set of weight vectors for scalarization
        # We use a grid in the simplex for weight vectors
        # For 3 objectives, we can use a grid of size k
        # Let's use a reasonably dense grid to cover the front well
        # Number of weights: ~20-30
        # We will use 25 weight vectors
        n_weights = 25
        weights = []
        # Generate points on the 2-simplex (3 objectives)
        # Using a simple grid approach
        # u, v in [0, 1] such that u+v <= 1
        # w0 = 1-u-v, w1=u, w2=v
        # To get uniform coverage, we can sample or use a grid
        # Grid:
        # Let's use a triangular grid
        # Number of points per side: 5 -> 5+4+3+2+1 = 15 points?
        # Or just random uniform on simplex
        rng = np.random.default_rng(42)
        
        # Generate random points on the simplex
        for _ in range(n_weights):
            # Dirichlet distribution with alpha=1 gives uniform distribution on simplex
            w = rng.dirichlet([1, 1, 1])
            weights.append(w)
        weights = np.array(weights)

        # 2. Allocate budget
        # We have 200 evaluations.
        # Each Nelder-Mead run uses (dim+1) initial points + iterations.
        # Dim=4, so 5 initial points.
        # If we run 25 weights, that's 125 initial points.
        # Remaining 75 for iterations?
        # Nelder-Mead can be expensive.
        # Alternative: Fewer weights, more iterations.
        # Let's use 10 weights.
        # 10 * 5 = 50 initial points.
        # Remaining 150 for iterations.
        # Each iteration is 1 eval.
        # So ~15 iterations per weight.
        # This is reasonable.
        
        n_weights = 10
        # Regenerate weights
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet([1, 1, 1])
            weights.append(w)
        weights = np.array(weights)

        # 3. Run local search for each weight
        # We will use a global list to store all evaluated points
        # and their objective values to compute hypervolume later
        # But the evaluator records EVERY returned objective vector.
        # So we just need to call problem(x) for every point we visit.
        # We don't need to store them explicitly for the return value,
        # but we might want to keep the best ones if we were returning them.
        # The return value is ignored.
        
        # However, we must be careful not to exceed budget.
        # We will track used evaluations.
        used = 0

        for w in weights:
            if time.time() > deadline:
                break
            if used >= budget:
                break

            # Define the scalarized objective
            def scalar_obj(x, w=w):
                nonlocal used
                if used >= budget:
                    return 0.0 # Should not happen if we break, but safety
                f = problem(x)
                used += 1
                return np.dot(w, f)

            # Initial point: random in bounds
            x0 = lo + (hi - lo) * rng.random(dim)
            
            # Nelder-Mead options
            # maxiter: limit iterations to save budget
            # We have roughly (budget - used) / n_weights left on average
            # Let's set maxiter to 15
            options = {
                'maxiter': 15,
                'xatol': 1e-6,
                'fatol': 1e-6
            }

            # Run Nelder-Mead
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', options=options)
            except Exception:
                continue

        # We have used 'used' evaluations.
        # If we have budget left, we can do a final random sampling or refine
        # But Nelder-Mead should have used most of it.
        
        # The problem statement says: "Every returned objective vector is recorded"
        # The evaluator calls problem(x) and records the result.
        # My scalar_obj calls problem(x) and increments used.
        # So all these calls are recorded.
        
        # Return something (ignored)
        return None
# EVOLVE-BLOCK-END
