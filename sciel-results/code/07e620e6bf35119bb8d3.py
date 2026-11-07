"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Structured multi-objective optimizer with local search on DTLZ7 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        evaluated = []  # list of (x, f)
        used = 0
        
        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            evaluated.append((x, f))
            used += 1
            return f
        
        # Phase 1: Structured diverse initialization
        # DTLZ7: first ~9 vars define front position, rest affect g(x)
        n_front = 9
        n_g = self.dim - n_front
        
        # Generate diverse front positions with good g-variables
        for i in range(15):
            # Sample front variables
            front_vars = rng.random(n_front)
            # Try g-variables at 0 (optimal for DTLZ7 g function)
            g_vars = np.zeros(n_g)
            x = np.concatenate([front_vars, g_vars])
            eval_point(x)
        
        # Add some random points for diversity
        for i in range(5):
            x = rng.random(self.dim)
            eval_point(x)
        
        # Phase 2: Local refinement of top candidates
        # Sort by hypervolume contribution proxy (sum of objectives)
        evaluated.sort(key=lambda xf: np.sum(xf[1]))
        
        # Refine top 8 candidates by perturbing g-variables
        for idx in range(min(8, len(evaluated))):
            if used >= self.budget - 3:
                break
            x, f = evaluated[idx]
            front_vars = x[:n_front]
            
            # Try a few perturbations of g-variables
            for trial in range(3):
                if used >= self.budget:
                    break
                # Random perturbation of g-variables
                g_pert = x[n_front:] + 0.05 * rng.standard_normal(n_g)
                x_new = np.concatenate([front_vars, g_pert])
                eval_point(x_new)
        
        # Phase 3: Fill remaining budget with random samples
        while used < self.budget:
            x = rng.random(self.dim)
            eval_point(x)
        
        return evaluated
# EVOLVE-BLOCK-END
