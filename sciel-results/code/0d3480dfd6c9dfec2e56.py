"""New program: Differential Evolution global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls_made = 0
        self.budget_exceeded = False
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal best_x, best_f
            if self.budget_exceeded:
                return 0.0
            self.calls_made += 1
            if self.calls_made > self.budget:
                self.budget_exceeded = True
                raise Exception("Budget Exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Differential Evolution is a population-based global optimizer.
        # It is distinct from local line-search methods like Powell.
        # We tune popsize to allow for a reasonable number of generations within the budget.
        # popsize=10 (2*dim) allows ~50 generations for 500 calls.
        
        try:
            res = differential_evolution(
                counted_func, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-10, 
                init='latinhypercube', 
                seed=42,
                polish=False # We handle polishing or rely on DE's convergence
            )
        except Exception:
            pass

        # If budget remains, do a final local polish with Nelder-Mead or Powell?
        # The directive says "do not reuse current designs". Current design is Multi-start Powell.
        # Adding a single Powell polish at the end might be considered reusing.
        # However, DE often leaves a small gap. A generic local polish is standard.
        # To be safe and distinct, I will stick to DE only or use a different local method if needed.
        # But DE with polish=False might stop early.
        # Let's allow DE to use its internal convergence.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
