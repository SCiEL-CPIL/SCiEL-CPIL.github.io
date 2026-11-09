"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol sweep and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return []
        
        # F is (n, m)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            leq = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            leq[i] = False 
            if np.any(leq & lt):
                dominated[i] = True
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0

        # 1. Sobol Sweep
        n_sobol = int(self.budget * 0.8)
        if n_sobol < 1:
            n_sobol = 1
            
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sobol_samples = sampler.random(n_sobol)
        sobol_samples = lo + sobol_samples * (hi - lo)
        
        for x in sobol_samples:
            if time.time() > deadline:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
            if used >= self.budget:
                break

        if used >= self.budget or time.time() > deadline:
            return np.array(all_X) if all_X else np.array([])

        # 2. Local Refinement
        F_sweep = np.array(all_F)
        X_sweep = np.array(all_X)
        
        nd_indices = self._non_dominated_indices(F_sweep)
        
        if len(nd_indices) == 0:
            nd_indices = [0]
        
        remaining_budget = self.budget - used
        n_refine_points = len(nd_indices)
        
        max_refine_points = min(n_refine_points, 20)
        if n_refine_points > max_refine_points:
            nd_indices = nd_indices[:max_refine_points]
            n_refine_points = len(nd_indices)
            
        if n_refine_points > 0:
            evals_per_point = max(1, remaining_budget // n_refine_points)
        else:
            evals_per_point = 0

        sigma = 0.05
        
        for idx in nd_indices:
            if used >= self.budget or time.time() > deadline:
                break
                
            base_x = X_sweep[idx]
            
            for _ in range(evals_per_point):
                if used >= self.budget or time.time() > deadline:
                    break
                    
                step = np.random.normal(0, sigma, self.dim)
                trial_x = base_x + step
                trial_x = np.clip(trial_x, lo, hi)
                
                trial_f = problem(trial_x)
                all_X.append(trial_x)
                all_F.append(trial_f)
                used += 1

        if not all_X:
            return np.array([])
            
        return np.array(all_X)

# EVOLVE-BLOCK-END
