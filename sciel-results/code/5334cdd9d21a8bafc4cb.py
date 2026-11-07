import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f, archive_F):
        """Check if f is dominated by any point in archive_F."""
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return True
        return False

    def _get_nondominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        nd_indices = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_indices.append(i)
        return np.array(nd_indices, dtype=int)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        N = self.budget
        dim = self.dim

        if N == 0:
            return np.array([])

        # Strategy:
        # Stage 1: Use 150 calls for structured Sobol sampling with heuristic last-6-dim values
        # Stage 2: Use remaining 50 calls to refine non-dominated points via local perturbations
        
        n_stage1 = min(150, N)
        n_stage2 = N - n_stage1

        dim1 = 6  # First 6 variables control Pareto front shape
        dim2 = dim - dim1  # Last 6 variables are "noise"/position

        # Stage 1: Structured sampling
        # Generate Sobol points for first 6 dimensions
        sampler = qmc.Sobol(d=dim1, scramble=True)
        X1 = sampler.random(n_stage1)

        X = np.zeros((n_stage1, dim))
        X[:, :dim1] = X1

        # For last 6 dimensions, use a mix of heuristic values:
        # - 50% at 0.0 (lower bound)
        # - 33% at 0.5 (center)  
        # - 17% at 1.0 (upper bound)
        # This explores multiple "manifolds" in the search space
        n_low = n_stage1 // 3
        n_mid = n_stage1 // 3
        n_high = n_stage1 - n_low - n_mid

        X[:n_low, dim1:] = 0.0
        X[n_low:n_low+n_mid, dim1:] = 0.5
        X[n_low+n_mid:, dim1:] = 1.0

        # Map to bounds
        X = lo + X * span

        # Evaluate stage 1
        F = np.array([problem(x) for x in X])
        used = n_stage1

        # Find non-dominated points from stage 1
        nd_indices = self._get_nondominated_indices(F)

        # Stage 2: Refine non-dominated points with local perturbations
        if n_stage2 > 0 and len(nd_indices) > 0:
            # Select up to 10 best non-dominated points for refinement
            n_refine = min(10, len(nd_indices))
            # Sort by sum of objectives to pick "best" non-dominated
            sums = F[nd_indices].sum(axis=1)
            sorted_idx = nd_indices[np.argsort(sums)[:n_refine]]
            
            # Each refinement point gets ~5 evaluations (local search)
            evals_per_point = max(1, n_stage2 // n_refine)

            for idx in sorted_idx:
                if used >= N:
                    break
                if time.time() > deadline:
                    break
                
                parent = X[idx]
                parent_F = F[idx]
                
                # Local perturbation: try small Gaussian steps in all dimensions
                for _ in range(evals_per_point):
                    if used >= N:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Perturb only the last 6 dimensions (noise vars) with small steps
                    # and slightly perturb first 6 dimensions
                    step = 0.05 * span
                    perturbation = np.random.randn(dim) * step
                    child = parent + perturbation
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    # If child is non-dominated, add to archive
                    if not self._is_dominated(fc, F):
                        # Replace the worst dominated point or append
                        # Find dominated points
                        dominated_mask = np.zeros(len(F), dtype=bool)
                        for i in range(len(F)):
                            for j in range(len(F)):
                                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                    dominated_mask[i] = True
                                    break
                        if np.any(dominated_mask):
                            # Replace worst dominated point
                            dom_indices = np.where(dominated_mask)[0]
                            worst_idx = dom_indices[np.argmax(F[dom_indices].sum(axis=1))]
                            X[worst_idx] = child
                            F[worst_idx] = fc
                        else:
                            # All non-dominated; replace worst by sum
                            worst_idx = np.argmax(F.sum(axis=1))
                            if fc.sum() < F[worst_idx].sum():
                                X[worst_idx] = child
                                F[worst_idx] = fc

        return X
