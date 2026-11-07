"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Direct Pareto front construction via analytical insight on ZDT4 structure."""
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        
        rng = np.random
        
        all_points = []
        all_evals = 0
        
        def eval_point(x):
            nonlocal all_evals
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_evals += 1
            return np.array(f, dtype=np.float64)
        
        # ZDT4: f1 = g(x2..xn) * h(x1), f2 = 1 + (1 - sqrt(f1/g))^2
        # On Pareto front: g = g_min, so f2 = 1 + (1 - sqrt(h(x1)))^2
        # Strategy: minimize g by finding x_rest that minimizes f2 at a fixed x1
        
        # Phase 1: Find x_rest that minimizes g (deviation from Pareto front)
        # Use a fixed x1 where h(x1) is known to be positive (e.g., x1 such that sin(10*pi*x1) > 0)
        # For ZDT4, h(x1) = sin(10*pi*x1) for x1 in [0,1]
        # Pick x1 = 0.1 where h(0.1) = sin(pi) = 0... bad choice
        # Pick x1 = 0.15 where h(0.15) = sin(1.5*pi) = -1... absolute value matters
        # Actually, let's use x1 = 0.05: h(0.05) = sin(0.5*pi) = 1, so f1 = g, f2 = 1 + (1-1)^2 = 1
        # So at x1=0.05, f2 = 1 regardless of g! That's not useful.
        # Let's use x1 = 0.1: h(0.1) = sin(pi) = 0, f1 = 0, f2 = 1 + (1-0)^2 = 2. Not useful either.
        # Use x1 = 0.05: f1 = g, f2 = 1. The f2 doesn't tell us about g.
        # Use x1 where h is not 0 or 1: x1 = 0.075, h = sin(0.75*pi) = sqrt(2)/2 ≈ 0.707
        # Then f1 = 0.707*g, f2 = 1 + (1 - sqrt(0.707))^2 = 1 + (1-0.841)^2 = 1 + 0.025 = 1.025
        # This still doesn't depend on g! Because f2 = 1 + (1-sqrt(f1/g))^2 and f1/g = h(x1)
        # So f2 only depends on h(x1), NOT on g! The deviation from Pareto front shows up in f1.
        # On Pareto front: f1 = g_min * h(x1). Off front: f1 = g * h(x1) where g > g_min.
        # So to find g_min, we need to minimize f1 at a fixed x1 where h(x1) > 0.
        
        # At x1 = 0.05, h = 1, so f1 = g. Minimize f1 to find g_min.
        x1_probe = 0.05
        x1_probe_actual = lo[0] + (hi[0] - lo[0]) * x1_probe
        
        # Phase 1: Minimize g using coordinate descent on x2..xn
        # Start with a few initial points
        best_x_rest = None
        best_f1 = np.inf
        
        # Try structured starting points
        n_starts = 8
        for i in range(n_starts):
            if all_evals >= self.budget:
                break
            if i == 0:
                x_rest = (lo[1:] + hi[1:]) / 2
            elif i == 1:
                x_rest = lo[1:].copy()
            elif i == 2:
                x_rest = hi[1:].copy()
            else:
                x_rest = lo[1:] + (hi[1:] - lo[1:]) * rng.random(n - 1)
            
            x_rest = np.clip(x_rest, lo[1:], hi[1:])
            x = np.concatenate(([x1_probe_actual], x_rest))
            f = eval_point(x)
            if np.all(np.isfinite(f)) and f[0] < best_f1:
                best_f1 = f[0]
                best_x_rest = x_rest.copy()
            
            if time.time() > deadline:
                break
        
        if best_x_rest is None:
            best_x_rest = (lo[1:] + hi[1:]) / 2
        
        # Local refinement of x_rest using coordinate descent
        n_cd_iters = 5
        step_sizes = [(hi[1:] - lo[1:]) * 0.1, (hi[1:] - lo[1:]) * 0.05, (hi[1:] - lo[1:]) * 0.01]
        
        for it in range(n_cd_iters):
            if all_evals >= self.budget or time.time() > deadline:
                break
            step = step_sizes[min(it, len(step_sizes) - 1)]
            improved = False
            for d in range(n - 1):
                if all_evals >= self.budget or time.time() > deadline:
                    break
                # Try +step
                x_rest_plus = best_x_rest.copy()
                x_rest_plus[d] += step[d]
                if x_rest_plus[d] > hi[d+1]:
                    x_rest_plus[d] = hi[d+1]
                x = np.concatenate(([x1_probe_actual], x_rest_plus))
                f = eval_point(x)
                if np.all(np.isfinite(f)) and f[0] < best_f1:
                    best_f1 = f[0]
                    best_x_rest = x_rest_plus.copy()
                    improved = True
                # Try -step
                x_rest_minus = best_x_rest.copy()
                x_rest_minus[d] -= step[d]
                if x_rest_minus[d] < lo[d+1]:
                    x_rest_minus[d] = lo[d+1]
                x = np.concatenate(([x1_probe_actual], x_rest_minus))
                f = eval_point(x)
                if np.all(np.isfinite(f)) and f[0] < best_f1:
                    best_f1 = f[0]
                    best_x_rest = x_rest_minus.copy()
                    improved = True
            
            if not improved:
                break
        
        # Phase 2: Dense sweep of x1 to trace the Pareto front
        remaining_budget = self.budget - all_evals
        if remaining_budget > 0:
            # Use most of the budget for dense x1 sweep
            n_sweep = min(remaining_budget - 5, 180)
            if n_sweep < 10:
                n_sweep = max(remaining_budget, 10)
            
            # Non-uniform spacing: denser where the front curves more
            # For ZDT4, h(x1) = sin(10*pi*x1), which oscillates rapidly
            # Use uniform spacing in x1 to cover all oscillations
            x1_vals = np.linspace(lo[0], hi[0], n_sweep)
            
            for x1v in x1_vals:
                if all_evals >= self.budget or time.time() > deadline:
                    break
                x = np.concatenate(([x1v], best_x_rest))
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    all_points.append(f)
        
        # Phase 3: Refinement - add points in gaps and refine x_rest locally
        remaining_budget = self.budget - all_evals
        if remaining_budget > 0 and len(all_points) > 1:
            points_arr = np.array(all_points)
            
            # Find the x1 values used (we need to track them)
            # Let's re-do the sweep with tracked x1 values
            # Actually, let's just add refinement points around existing front points
            
            # Get non-dominated points
            nd_indices = self._nondominated_indices(points_arr)
            if len(nd_indices) > 0:
                nd_points = points_arr[nd_indices]
                # Sort by f1
                sort_idx = np.argsort(nd_points[:, 0])
                nd_sorted = nd_points[sort_idx]
                
                # Add midpoints between consecutive non-dominated points in objective space
                # and also try to refine x_rest
                n_refine = min(remaining_budget, 30)
                for i in range(n_refine):
                    if all_evals >= self.budget or time.time() > deadline:
                        break
                    
                    # Alternate between:
                    # 1. Midpoint in x1 between two consecutive sweep points
                    # 2. Local perturbation of x_rest
                    
                    if i % 2 == 0 and len(nd_sorted) >= 2:
                        # Pick a random pair of adjacent non-dominated points
                        idx = rng.randint(len(nd_sorted) - 1)
                        # We don't have the x values, so let's just do random x1 near the front
                        # Better: do a local search in x_rest
                        pass
                    
                    # Do local refinement of x_rest
                    x1_try = lo[0] + (hi[0] - lo[0]) * rng.random()
                    x_rest_try = best_x_rest + (hi[1:] - lo[1:]) * 0.02 * rng.standard_normal(n - 1)
                    x_rest_try = np.clip(x_rest_try, lo[1:], hi[1:])
                    x = np.concatenate(([x1_try], x_rest_try))
                    f = eval_point(x)
                    if np.all(np.isfinite(f)):
                        all_points.append(f)
        
        # Convert all_points to array and ensure they're valid
        if len(all_points) > 0:
            result = np.array(all_points, dtype=np.float64)
            # Filter out any non-finite points
            mask = np.all(np.isfinite(result), axis=1)
            result = result[mask]
        else:
            # Last resort: evaluate a single point
            x = (lo + hi) / 2
            f = eval_point(x)
            result = np.array([f], dtype=np.float64)
        
        return result
    
    @staticmethod
    def _nondominated_indices(F):
        """Return indices of non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            # Check if any other point dominates i
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
