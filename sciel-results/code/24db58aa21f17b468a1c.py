"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, rng, deadline):
        """Run Nelder-Mead from x0 with given budget."""
        n = self.dim
        # Build initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.5
            simplex.append(np.clip(xi, -5, 5))
        
        fvals = [func(x) for x in simplex]
        calls = n + 1
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = min(fvals)
        
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        while calls < budget and time.time() < deadline:
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if time.time() > deadline:
                break
            
            # Check convergence
            fspread = fvals[-1] - fvals[0]
            if fspread < 1e-10:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls += 1
            if calls >= budget or time.time() > deadline:
                break
            
            if fr < fvals[0]:
                # Try expansion
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break
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
                    xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                else:
                    # Inside contraction
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                fc = func(xc)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break
                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget or time.time() > deadline:
                            break
                    if calls >= budget or time.time() > deadline:
                        break
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Strategy: multiple Nelder-Mead starts with different initial points
        # Budget 500, dim 5. Each NM start needs ~6 initial + iterations.
        # With 500 budget, we can do several restarts.
        
        # Phase 1: Broad exploration with a few coarse restarts
        # Use stratified initial points for diversity
        n_starts = 5
        budget_per_start = self.budget // n_starts  # 100 each
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(self.dim))
        # Random points in [-4, 4]
        for _ in range(n_starts - 1):
            sp = rng.uniform(-4, 4, self.dim)
            start_points.append(sp)
        
        for i, sp in enumerate(start_points):
            remaining_budget = self.budget - total_calls
            if remaining_budget < 10:
                break
            if time.time() > deadline:
                break
            
            b = min(budget_per_start, remaining_budget)
            bx, bf, used = self._nelder_mead(func, sp, b, rng, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx
        
        # Phase 2: Fine-tune around best with remaining budget
        if best_x is not None and total_calls < self.budget and time.time() < deadline:
            remaining = self.budget - total_calls
            if remaining > 10:
                # Run another NM from the best point found
                bx, bf, used = self._nelder_mead(func, best_x, remaining, rng, deadline)
                total_calls += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx
        
        if best_x is None:
            # Fallback: just return a point
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
