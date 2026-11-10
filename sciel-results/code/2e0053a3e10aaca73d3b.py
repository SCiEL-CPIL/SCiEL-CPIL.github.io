"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling + multi-start Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0

    def _call_func(self, func, x):
        """Call function and track budget."""
        if self.calls_made >= self.budget:
            return np.inf
        self.calls_made += 1
        try:
            return func(x)
        except Exception:
            return np.inf

    def _nelder_mead(self, func, x0, lower=-5.0, upper=5.0, max_iter=200):
        """
        Robust Nelder-Mead implementation tuned for ill-conditioned functions.
        Returns (best_x, best_f).
        """
        dim = self.dim
        
        # Initialize simplex with adaptive step size
        # For ill-conditioned functions, use smaller steps to avoid overshooting
        step = 0.5
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            # Ensure we don't start outside the box
            if xi[i] > upper:
                xi[i] = upper - step if upper - step > lower else upper
            if xi[i] < lower:
                xi[i] = lower + step if lower + step < upper else lower
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = []
        for v in simplex:
            fv = self._call_func(func, v)
            fvals.append(fv)
            
        # Check if we ran out of budget immediately
        if self.calls_made >= self.budget:
            best_idx = np.argmin(fvals)
            return simplex[best_idx].copy(), fvals[best_idx]

        # Nelder-Mead parameters tuned for ill-conditioned functions
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma = 0.5  # shrink
        
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = np.min(fvals)
        
        # Iteration loop
        for _ in range(max_iter):
            if self.calls_made >= self.budget:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            # Convergence check - use relative tolerance
            if abs(fvals[-1] - fvals[0]) < 1e-10 * (1 + abs(fvals[0])):
                break
                
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, lower, upper)
            
            if self.calls_made >= self.budget:
                break
            f_r = self._call_func(func, x_r)
            
            if f_r < fvals[0]:
                # Try expansion
                if self.calls_made < self.budget:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lower, upper)
                    f_e = self._call_func(func, x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                
                x_c = np.clip(x_c, lower, upper)
                
                if self.calls_made >= self.budget:
                    break
                f_c = self._call_func(func, x_c)
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        
                        if self.calls_made >= self.budget:
                            break
                        fvals[i] = self._call_func(func, simplex[i])
                    
                    if self.calls_made >= self.budget:
                        break

            if self.calls_made >= self.budget:
                break

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        self.calls_made = 0

        # Allocate budget: 30% for global exploration, 70% for local refinement
        n_sobol = min(150, max(50, int(budget * 0.3)))
        
        # 1. Global Exploration with Sobol Sequence
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * (upper - lower) + lower

        # Evaluate all Sobol points
        sobol_fvals = []
        for i in range(n_sobol):
            if self.calls_made >= budget:
                break
            x = sobol_points[i]
            fx = self._call_func(func, x)
            sobol_fvals.append(fx)
        
        if self.calls_made >= budget or not sobol_fvals:
            # Return best from Sobol if no budget left
            if sobol_fvals:
                best_idx = np.argmin(sobol_fvals)
                return sobol_points[best_idx].copy(), sobol_fvals[best_idx]
            return np.zeros(dim), np.inf

        # Sort by function value
        sorted_indices = np.argsort(sobol_fvals)
        
        # 2. Multi-start Nelder-Mead from top candidates
        # Use remaining budget for local refinement
        remaining_budget = budget - self.calls_made
        n_starts = min(5, len(sorted_indices))  # Try up to 5 best starts
        
        best_x = sobol_points[sorted_indices[0]].copy()
        best_f = sobol_fvals[sorted_indices[0]]
        
        for s in range(n_starts):
            if self.calls_made >= budget:
                break
            
            start_idx = sorted_indices[s]
            start_x = sobol_points[start_idx].copy()
            
            # Allocate remaining budget proportionally to remaining starts
            # Give more budget to earlier starts
            frac = 1.0 / (n_starts - s)
            max_iter = max(10, int(remaining_budget * frac * 0.8))
            
            x_local, f_local = self._nelder_mead(func, start_x, lower, upper, max_iter=max_iter)
            
            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
