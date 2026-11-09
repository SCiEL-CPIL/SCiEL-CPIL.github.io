import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using multi-start Nelder-Mead on scalarized objectives."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # All evaluated points
        all_X = []
        all_F = []
        eval_count = 0

        def eval_point(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            eval_count += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # Strategy: Use weighted-sum scalarization with different weight vectors
        # Each Nelder-Mead run uses a fixed weight vector w, minimizing w·f(x)
        # We'll use multiple weight vectors to cover the Pareto front

        # Generate weight vectors using simplex-lattice or random
        n_weights = 8
        # Simplex-lattice for 3 objectives: weights on 3D simplex
        # For n_obj=3, we can use a grid
        weights = []
        # Use a stratified set of weights
        for i in range(n_weights):
            # Random weights on simplex
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        
        # Also add extreme points (unit vectors) for coverage
        for k in range(n_obj):
            w = np.zeros(n_obj)
            w[k] = 1.0
            weights.append(w)
        
        n_total_weights = len(weights)
        
        # Budget per weight: distribute remaining budget
        # First, do a small random initialization
        n_init = min(10, self.budget // 4)
        for i in range(n_init):
            x = lo + span * rng.random(dim)
            eval_point(x)
        
        # Now run Nelder-Mead for each weight vector
        # Each NM run has a budget of about (budget - n_init) / n_total_weights
        budget_per_nm = max(5, (self.budget - n_init) // n_total_weights)
        
        for w_idx, w in enumerate(weights):
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Start from a random point or from best found so far for this weight
            # Use random start for diversity
            x0 = lo + span * rng.random(dim)
            
            def scalarized(x):
                if time.time() > deadline:
                    return 1e10
                f = np.array(problem(x))
                eval_count += 1
                all_X.append(np.clip(x, lo, hi).copy())
                all_F.append(f)
                return float(np.dot(w, f))
            
            # Run Nelder-Mead with limited function evaluations
            try:
                result = minimize(
                    scalarized, x0, method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': budget_per_nm, 'maxfev': budget_per_nm, 'xatol': 1e-4, 'fatol': 1e-4}
                )
            except Exception:
                pass
        
        # Return all evaluated objective vectors
        if all_F:
            return np.array(all_F)
        return np.array([])
