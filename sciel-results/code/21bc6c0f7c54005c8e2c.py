import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using Sobol exploration + local Gaussian refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Phase 1: Sobol quasi-random exploration
        # Use about 60% of budget for initial spread
        n_sobol = int(self.budget * 0.6)
        n_sobol = max(20, n_sobol)
        
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        X_sobol = lo + sobol.random(n_sobol) * range_

        all_X = []
        all_F = []
        eval_count = 0

        for x in X_sobol:
            if eval_count >= self.budget or time.time() > deadline - 5:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            eval_count += 1

        # Phase 2: Local Gaussian refinement around non-dominated points
        if eval_count < self.budget and time.time() < deadline - 10:
            F_array = np.array(all_F)
            nd_indices = self._non_dominated_indices(F_array)
            
            if len(nd_indices) > 0:
                # Sort non-dominated points by their sum of objectives (prefer better ones)
                nd_sums = np.sum(F_array[nd_indices], axis=1)
                nd_sorted = nd_indices[np.argsort(nd_sums)]
                
                # Allocate remaining budget to refine around best non-dominated points
                remaining_budget = self.budget - eval_count
                n_refine_points = min(len(nd_sorted), max(1, remaining_budget // 10))
                
                # Gaussian perturbation scale
                sigma = 0.15  # Fraction of range
                
                for idx in nd_sorted[:n_refine_points]:
                    if eval_count >= self.budget or time.time() > deadline - 5:
                        break
                    
                    center = all_X[idx]
                    # Number of perturbations per center
                    n_perturb = max(1, (self.budget - eval_count) // max(1, len(nd_sorted) - (nd_sorted.tolist().index(idx))))
                    n_perturb = min(n_perturb, 20)
                    
                    for _ in range(n_perturb):
                        if eval_count >= self.budget or time.time() > deadline - 5:
                            break
                        
                        # Gaussian perturbation
                        perturb = np.random.normal(0, sigma, size=self.dim)
                        x_new = center + perturb * range_
                        x_new = np.clip(x_new, lo, hi)
                        
                        f_new = problem(x_new)
                        all_X.append(x_new)
                        all_F.append(f_new)
                        eval_count += 1

        return np.array(all_X)
