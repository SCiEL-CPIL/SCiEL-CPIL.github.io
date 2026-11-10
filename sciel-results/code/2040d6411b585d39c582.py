"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling + robust Nelder-Mead local refinement."""

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

    def _nelder_mead(self, func, x0, lower=-5.0, upper=5.0):
        """
        Robust Nelder-Mead implementation optimized for ill-conditioned functions.
        Returns (best_x, best_f).
        """
        dim = self.dim
        
        # Initialize simplex with adaptive step size
        # For bent cigar, use larger step in the ill-conditioned direction
        step = 1.5
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            if xi[i] > upper:
                xi[i] = upper - step if upper - step > lower else upper
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = []
        for v in simplex:
            fv = self._call_func(func, v)
            fvals.append(fv)
            
        if self.calls_made >= self.budget:
            best_idx = np.argmin(fvals)
            return simplex[best_idx].copy(), fvals[best_idx]

        # Nelder-Mead parameters tuned for ill-conditioned problems
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = np.min(fvals)
        
        # Iteration loop with convergence detection
        for _ in range(150):
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
                
            # Convergence check: relative spread in function values
            f_spread = abs(fvals[-1] - fvals[0])
            f_scale = max(1.0, abs(fvals[0]))
            if f_spread < 1e-10 * f_scale:
                break
            
            # Also check if simplex has collapsed
            centroid = np.mean(simplex[:-1], axis=0)
            max_dist = max(np.linalg.norm(v - centroid) for v in simplex)
            if max_dist < 1e-8:
                break
                
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

        # 1. Deterministic Sobol sampling for systematic coverage
        # Use 20 points as specified in the research programme
        n_sobol = 20
        sampler = qmc.Sobol(d=dim, scramble=False, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * (upper - lower) + lower

        best_x = None
        best_f = np.inf

        for i in range(n_sobol):
            if self.calls_made >= budget:
                break
            x = sobol_points[i]
            fx = self._call_func(func, x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # 2. Local refinement with Nelder-Mead on the best candidate
        if self.calls_made < budget and best_x is not None:
            best_x, best_f = self._nelder_mead(func, best_x, lower, upper)
            
            # 3. If budget remains, try a second refinement pass with smaller step
            if self.calls_made < budget and self.budget - self.calls_made > 10:
                # Create a tighter simplex around the current best
                small_step = 0.3
                simplex2 = [best_x.copy()]
                for i in range(dim):
                    xi = best_x.copy()
                    xi[i] += small_step
                    if xi[i] > upper:
                        xi[i] = upper - small_step if upper - small_step > lower else upper
                    simplex2.append(xi)
                
                # Evaluate this tighter simplex
                fvals2 = []
                for v in simplex2:
                    if self.calls_made >= budget:
                        break
                    fv = self._call_func(func, v)
                    fvals2.append(fv)
                
                if len(fvals2) == dim + 1 and self.calls_made < budget:
                    # Run a short Nelder-Mead on this tighter simplex
                    order = np.argsort(fvals2)
                    simplex2 = [simplex2[i] for i in order]
                    fvals2 = [fvals2[i] for i in order]
                    
                    if fvals2[0] < best_f:
                        best_f = fvals2[0]
                        best_x = simplex2[0].copy()
                    
                    # Quick refinement iterations
                    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                    for _ in range(30):
                        if self.calls_made >= budget:
                            break
                        
                        order = np.argsort(fvals2)
                        simplex2 = [simplex2[i] for i in order]
                        fvals2 = [fvals2[i] for i in order]
                        
                        if fvals2[0] < best_f:
                            best_f = fvals2[0]
                            best_x = simplex2[0].copy()
                        
                        f_spread = abs(fvals2[-1] - fvals2[0])
                        f_scale = max(1.0, abs(fvals2[0]))
                        if f_spread < 1e-10 * f_scale:
                            break
                        
                        centroid = np.mean(simplex2[:-1], axis=0)
                        x_r = centroid + alpha * (centroid - simplex2[-1])
                        x_r = np.clip(x_r, lower, upper)
                        
                        if self.calls_made >= budget:
                            break
                        f_r = self._call_func(func, x_r)
                        
                        if f_r < fvals2[0]:
                            if self.calls_made < budget:
                                x_e = centroid + gamma * (x_r - centroid)
                                x_e = np.clip(x_e, lower, upper)
                                f_e = self._call_func(func, x_e)
                                if f_e < f_r:
                                    simplex2[-1] = x_e
                                    fvals2[-1] = f_e
                                else:
                                    simplex2[-1] = x_r
                                    fvals2[-1] = f_r
                            else:
                                simplex2[-1] = x_r
                                fvals2[-1] = f_r
                        elif f_r < fvals2[-2]:
                            simplex2[-1] = x_r
                            fvals2[-1] = f_r
                        else:
                            if f_r < fvals2[-1]:
                                x_c = centroid + rho * (x_r - centroid)
                            else:
                                x_c = centroid - rho * (simplex2[-1] - centroid)
                            x_c = np.clip(x_c, lower, upper)
                            
                            if self.calls_made >= budget:
                                break
                            f_c = self._call_func(func, x_c)
                            
                            if f_c < min(f_r, fvals2[-1]):
                                simplex2[-1] = x_c
                                fvals2[-1] = f_c
                            else:
                                for i in range(1, len(simplex2)):
                                    simplex2[i] = simplex2[0] + sigma * (simplex2[i] - simplex2[0])
                                    simplex2[i] = np.clip(simplex2[i], lower, upper)
                                    if self.calls_made >= budget:
                                        break
                                    fvals2[i] = self._call_func(func, simplex2[i])
                                if self.calls_made >= budget:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
