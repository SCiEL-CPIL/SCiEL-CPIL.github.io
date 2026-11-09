"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol quasi-random initialization with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use Sobol sequence for initial space-filling design
        # scipy.stats.qmc.Sobol is available
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        
        # Determine how many points to use for initial sweep
        # Use about 60% of budget for initial exploration
        n_initial = int(0.6 * self.budget)
        # Sobol requires power of 2 for best performance, but can generate any number
        n_sobol = 1
        while n_sobol < n_initial:
            n_sobol *= 2
        sobol_points = sobol.random(n_sobol)[:n_initial]
        
        # Map to bounds
        X_init = lo + (hi - lo) * sobol_points
        F_init = np.array([problem(x) for x in X_init])
        used = n_initial
        
        # Find non-dominated points
        def find_nondominated_indices(f):
            n = len(f)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                        nd[i] = False
            return np.where(nd)[0]
        
        # Local refinement around non-dominated points
        nd_idx = find_nondominated_indices(F_init)
        if len(nd_idx) == 0:
            nd_idx = np.array([0])
        
        # Adaptive mutation scale
        sigma = 0.05 * (hi - lo)
        
        # Use remaining budget for local perturbations around non-dominated points
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select a non-dominated point to perturb
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X_init[parent_idx].copy()
            
            # Gaussian perturbation
            child = parent + sigma * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Add to population if it improves diversity
            # Simple check: if child is not dominated by any existing point, add it
            # To keep memory bounded, we'll track a growing list of good points
            if not hasattr(self, '_refined_X'):
                self._refined_X = [X_init.copy()]
                self._refined_F = [F_init.copy()]
            
            # Check if child is dominated by any point in current refined set
            dominated = False
            for f_arr in self._refined_F:
                for fi in f_arr:
                    if np.all(fi <= fc) and np.any(fi < fc):
                        dominated = True
                        break
                if dominated:
                    break
            
            if not dominated:
                self._refined_X[-1] = np.vstack([self._refined_X[-1], child.reshape(1, -1)])
                self._refined_F[-1] = np.vstack([self._refined_F[-1], fc.reshape(1, -1)])
            
            # Update non-dominated set periodically
            if used % 20 == 0:
                all_F = np.vstack(self._refined_F)
                nd_idx = find_nondominated_indices(all_F)
                # Get the non-dominated points from the last chunk
                last_F = self._refined_F[-1]
                last_nd = [i for i in range(len(last_F)) if any(np.all_equal(last_F[i], all_F[j]) for j in nd_idx)]
                if len(last_nd) > 0:
                    nd_idx = last_nd
                else:
                    # Fall back to global non-dominated indices mapped to local
                    nd_idx = np.array([0])
        
        return X_init
# EVOLVE-BLOCK-END
