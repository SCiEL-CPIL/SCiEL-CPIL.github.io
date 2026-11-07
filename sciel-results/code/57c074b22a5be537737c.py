"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Grid-free continuous multi-objective optimization for ZDT1.
    
    Strategy: 
    1. Select K target points along the Pareto front (x1 values).
    2. For each target, initialize x with x1=target and x[1:]=0.5.
    3. Use remaining budget to locally minimize f2 (by optimizing x[1:]) for each target,
       ensuring g(x) is minimized.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We want to cover the Pareto front well.
        # ZDT1 front: f1 in [0,1].
        # Let's pick 20 targets.
        n_targets = 20
        # Ensure we don't exceed budget
        if self.budget < n_targets:
            n_targets = max(2, self.budget // 2)
            
        # Targets for x1 (which is f1)
        # Spread evenly from lo[0] to hi[0]
        x1_targets = np.linspace(lo[0], hi[0], n_targets)
        
        archive = []
        used = 0
        
        # Initialize points
        for t in x1_targets:
            x = np.zeros(self.dim)
            x[0] = t
            # For ZDT1, x[1:] = 0.5 minimizes g(x)
            # If bounds are not symmetric, we might need to adjust, but 0.5 is usually optimal for standard ZDT1
            # Let's start with 0.5, clipped
            x[1:] = 0.5
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            archive.append({'x': x.copy(), 'f': f.copy(), 'target': t})
            used += 1
            
        if used >= self.budget:
            return np.array([a['x'] for a in archive])

        # Local refinement:
        # For each point, we want to minimize f2.
        # f2 = g(x) * (1 - sqrt(x1/g(x)))
        # Since x1 is fixed (target), minimizing f2 is equivalent to minimizing g(x) (approximately, for small g).
        # Actually, if g(x) is minimized at 1, f2 is minimized.
        # We will perform a local search on x[1:] to minimize f2, keeping x[0] fixed.
        
        remaining_budget = self.budget - used
        # Distribute remaining budget among targets
        # More budget for points that might be worse? Or just equal?
        # Let's do equal distribution.
        evals_per_target = max(1, remaining_budget // n_targets)
        
        for i in range(n_targets):
            if used >= self.budget:
                break
                
            best_x = archive[i]['x'].copy()
            best_f = archive[i]['f'].copy()
            
            # Simple local search: random perturbations of x[1:]
            # We keep x[0] fixed to maintain the target f1
            for _ in range(evals_per_target):
                if used >= self.budget:
                    break
                    
                # Generate candidate
                x_new = best_x.copy()
                # Perturb x[1:]
                # Step size: small
                step_size = 0.1
                noise = np.random.normal(0, step_size, self.dim - 1)
                x_new[1:] += noise
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Accept if f2 is lower (f1 is same since x[0] is same)
                if f_new[1] < best_f[1]:
                    best_x = x_new.copy()
                    best_f = f_new.copy()
                    archive[i]['x'] = best_x
                    archive[i]['f'] = best_f
            
        return np.array([a['x'] for a in archive])
# EVOLVE-BLOCK-END
