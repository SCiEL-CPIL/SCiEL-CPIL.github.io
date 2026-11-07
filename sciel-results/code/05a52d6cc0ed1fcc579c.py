"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with Nelder-Mead polish and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget_left, deadline, rng):
        """Custom Nelder-Mead implementation with budget awareness."""
        dim = self.dim
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            if abs(xi[i]) > 1e-8:
                xi[i] *= 1.05
            else:
                xi[i] += 0.05
            simplex.append(xi)
        
        fvals = [func(x) for x in simplex]
        calls_used = dim + 1
        remaining = budget_left - calls_used
        
        if remaining <= 0:
            return x0, min(fvals)
        
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-12 * (1 + abs(fvals[0])):
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1
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
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_used += 1
                        if calls_used >= budget_left:
                            break
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_used += 1
                        if calls_used >= budget_left:
                            break
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        
        # Multi-start: try several random initial points
        n_starts = min(10, budget // 50)
        best_x = None
        best_f = np.inf
        
        for s in range(n_starts):
            if time.time() > deadline - 5:
                break
            
            # Random start, biased towards center for ridge functions
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-2, 2, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            x0 = np.clip(x0, -5, 5)
            fx0 = func(x0)
            
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            # Use (1+1)-ES for a short run to find a good local minimum
            x = x0.copy()
            fx = fx0
            sigma = 1.0
            es_budget = min(30, budget - (budget // n_starts) * s - 10)
            
            for _ in range(es_budget):
                if time.time() > deadline - 5:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Polish the best found point with Nelder-Mead
        remaining_budget = budget - (budget // n_starts) * n_starts - 30 * n_starts
        remaining_budget = max(remaining_budget, 20)
        
        if best_x is not None and remaining_budget > 10:
            polished_x, polished_f = self._nelder_mead(
                func, best_x, remaining_budget, deadline, rng
            )
            if polished_f < best_f:
                best_f = polished_f
                best_x = polished_x
        
        return best_x, best_f
# EVOLVE-BLOCK-END
