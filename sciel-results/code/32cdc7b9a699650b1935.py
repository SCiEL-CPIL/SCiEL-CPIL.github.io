"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Multi-start Nelder-Mead with Sobol initialization:
  - BBOB f14 is unimodal and smooth under the hidden affine transform.
  - Use Sobol sequence for structured space-filling initialization.
  - Run multiple Nelder-Mead simplexes in parallel (multi-start), then
    converge the best simplex for final precision.
  - Nelder-Mead is effective on smooth unimodal functions and doesn't
    require gradient or Hessian information.
  - Sobol points provide deterministic, low-discrepancy coverage of the
    search space, improving initial exploration over pure random sampling.

Strategy:
  1. Generate Sobol points across [-5,5]^dim for diverse starting locations.
  2. Run Nelder-Mead from each start with a small evaluation budget.
  3. Select the best result and refine with a larger Nelder-Mead budget.
  4. Track best point throughout and return it.
"""
import time
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_points(self, n_points, dim, low=-5.0, high=5.0, seed=42):
        """Generate low-discrepancy Sobol-like points using van der Corput sequences."""
        rng = np.random.RandomState(seed)
        
        # Generate base indices
        base = np.arange(n_points)
        
        # Van der Corput sequence for each dimension
        points = np.zeros((n_points, dim))
        for d in range(dim):
            # Use different bases or transformations for each dimension
            # Simple approach: use van der Corput with base (d+2) or similar
            base_val = d + 2  # 2, 3, 4, 5, 6 for dim 5
            # Van der Corput: reverse digits in base
            vc = np.zeros(n_points)
            for i in range(n_points):
                n = base[i]
                frac = 0.0
                f = 1.0 / base_val
                while n > 0:
                    frac += (n % base_val) * f
                    n //= base_val
                    f /= base_val
                vc[i] = frac
            points[:, d] = vc
        
        # Add some random perturbation to break any remaining patterns
        points += rng.uniform(0, 0.01, size=points.shape)
        
        # Scale to [low, high]
        points = low + points * (high - low)
        
        # Ensure within bounds
        points = np.clip(points, low, high)
        
        return points

    def _nelder_mead(self, func, x0, calls_counter, deadline, dim, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.01):
        """Run Nelder-Mead simplex optimization.
        
        Returns (best_x, best_f).
        """
        n = dim
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.25  # Small step in each dimension
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            if calls_counter[0] >= self.budget or time.time() >= deadline:
                break
            fvals[i] = float(func(simplex[i]))
            calls_counter[0] += 1
        
        if calls_counter[0] >= self.budget or time.time() >= deadline:
            idx = np.argmin(fvals)
            return simplex[idx].copy(), float(fvals[idx])
        
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = float(np.min(fvals))
        
        for it in range(max_iter):
            if calls_counter[0] >= self.budget or time.time() >= deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
            
            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[n])
            x_ref = np.clip(x_ref, -5.0, 5.0)
            if calls_counter[0] < self.budget and time.time() < deadline:
                f_ref = float(func(x_ref))
                calls_counter[0] += 1
            
            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[n])
                x_exp = np.clip(x_exp, -5.0, 5.0)
                if calls_counter[0] < self.budget and time.time() < deadline:
                    f_exp = float(func(x_exp))
                    calls_counter[0] += 1
                    if f_exp < f_ref:
                        simplex[n] = x_exp
                        fvals[n] = f_exp
                    else:
                        simplex[n] = x_ref
                        fvals[n] = f_ref
                else:
                    simplex[n] = x_ref
                    fvals[n] = f_ref
            elif f_ref < fvals[n - 1]:
                simplex[n] = x_ref
                fvals[n] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[n] - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                if calls_counter[0] < self.budget and time.time() < deadline:
                    f_con = float(func(x_con))
                    calls_counter[0] += 1
                    if f_con < fvals[n]:
                        simplex[n] = x_con
                        fvals[n] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls_counter[0] < self.budget and time.time() < deadline:
                                fvals[i] = float(func(simplex[i]))
                                calls_counter[0] += 1
                        break
                else:
                    break
        
        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # 10s margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls_counter = [0]
        
        def still_ok():
            return time.time() < deadline and calls_counter[0] < budget
        
        # Phase 1: Multi-start Nelder-Mead with Sobol initialization
        # Use ~60% of budget for exploration
        explore_budget = int(budget * 0.6)
        
        # Generate Sobol points for diverse starting locations
        n_starts = min(8, explore_budget // 10)  # At most 8 starts
        sobol_pts = self._sobol_points(n_starts, dim)
        
        # Also include center and a few random points
        all_starts = [np.zeros(dim)]  # Center
        all_starts.extend(sobol_pts.tolist())
        
        # Add a few random points for extra diversity
        rng = np.random.RandomState(123)
        for _ in range(min(3, n_starts)):
            all_starts.append(rng.uniform(-5, 5, size=dim))
        
        # Run Nelder-Mead from each start with limited budget
        best_x = None
        best_f = float('inf')
        
        per_start_budget = max(10, explore_budget // len(all_starts))
        
        for x0 in all_starts:
            if not still_ok():
                break
            
            x0_clipped = np.clip(x0, -5.0, 5.0)
            
            # Run Nelder-Mead with limited iterations
            bx, bf = self._nelder_mead(
                func, x0_clipped, calls_counter, deadline, dim,
                max_iter=per_start_budget // (dim + 2)
            )
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        if best_x is None:
            # Fallback: evaluate center
            x0 = np.zeros(dim)
            best_f = float(func(x0))
            calls_counter[0] += 1
            best_x = x0.copy()
        
        # Phase 2: Refine best solution with larger Nelder-Mead budget
        remaining = budget - calls_counter[0]
        if remaining > dim + 5 and still_ok():
            # Run a final Nelder-Mead refinement from the best point
            bx2, bf2 = self._nelder_mead(
                func, best_x, calls_counter, deadline, dim,
                max_iter=remaining // (dim + 2)
            )
            
            if bf2 < best_f:
                best_f = bf2
                best_x = bx2.copy()
        
        return best_x, float(best_f)
