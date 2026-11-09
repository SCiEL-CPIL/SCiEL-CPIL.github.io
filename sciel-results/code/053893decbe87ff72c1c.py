"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Custom Nelder-Mead Optimizer with anisotropy-aware simplex."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_iter=200, ftol=1e-10, xatol=1e-8):
        """Custom Nelder-Mead with adaptive simplex scaling for anisotropic functions."""
        n = self.dim
        # Initialize simplex: one point at x0, others offset along axes
        # Use larger initial step for Bent Cigar (high anisotropy)
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.5  # Initial step size
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            fvals[i] = func(simplex[i])
        
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]
        
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrink
        
        for _ in range(max_iter):
            # Check convergence
            f_range = np.max(fvals) - np.min(fvals)
            if f_range < ftol:
                break
            x_range = np.max(np.abs(simplex - simplex[0]))
            if x_range < xatol:
                break
            
            # Order simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                fc = func(xc)
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
            
            # Track best
            cur_best_idx = np.argmin(fvals)
            if fvals[cur_best_idx] < best_f:
                best_f = fvals[cur_best_idx]
                best_x = simplex[cur_best_idx].copy()
        
        return best_x, best_f

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 325.0  # 300s limit, 25s margin
        
        # Wrapper to count calls and enforce limits
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        best_x = np.zeros(self.dim)
        best_fx = float('inf')
        
        # Better initial points: center + structured random + axis-aligned
        rng = np.random.default_rng(123)
        starts = [np.zeros(self.dim)]
        # Add points along axes to help with bent cigar (elongated valley)
        for axis in range(self.dim):
            for sign in [1, -1]:
                pt = np.zeros(self.dim)
                pt[axis] = sign * 3.0
                starts.append(pt)
        # Add some random points
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Phase 1: Multi-start Nelder-Mead with moderate tolerances
        for x0 in starts:
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
                
            try:
                x_res, f_res = self._nelder_mead(
                    eval_func,
                    x0,
                    max_iter=150,
                    ftol=1e-12,
                    xatol=1e-8
                )
                if f_res < best_fx:
                    best_fx = f_res
                    best_x = x_res.copy()
            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: Final high-precision polish from the best point found
        if self.calls < self.budget and time.time() - self.start_time < self.time_limit - 10:
            try:
                x_res, f_res = self._nelder_mead(
                    eval_func,
                    best_x,
                    max_iter=300,
                    ftol=1e-14,
                    xatol=1e-10
                )
                if f_res < best_fx:
                    best_fx = f_res
                    best_x = x_res.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Ensure we return something valid
        if best_x is None or not np.all(np.isfinite(best_x)):
             best_x = np.zeros(self.dim)
             if self.calls < self.budget:
                 best_fx = eval_func(best_x)
             else:
                 best_fx = 0.0
                 
        return best_x, best_fx
# EVOLVE-BLOCK-END
