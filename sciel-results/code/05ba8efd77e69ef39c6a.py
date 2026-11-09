# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured search exploiting WFG9's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        range_ = hi - lo
        
        # Track all evaluated points and their objective values
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        def is_time_up():
            return time.time() > deadline

        # Phase 1: Dense deterministic sweep along the primary decision variables
        # WFG9's Pareto front is controlled primarily by x[0], x[1], x[2]
        # We sweep these three variables while optimizing the rest locally
        
        # First, find a good base point by random sampling
        base_X = lo + range_ * rng.random(dim)
        base_F = evaluate(base_X)
        
        # Try to improve the base point using a few random restarts
        for _ in range(10):
            if is_time_up() or used >= budget:
                break
            cand = lo + range_ * rng.random(dim)
            cf = evaluate(cand)
            if np.sum(cf) < np.sum(base_F):
                base_X = cand.copy()
                base_F = cf.copy()
        
        # Now sweep along the first three dimensions
        # Use a grid of 5 points per dimension for the primary variables
        # That's 5^3 = 125 evaluations for the sweep
        sweep_points = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if used >= budget or is_time_up():
                        break
                    x = base_X.copy()
                    x[0] = lo[0] + (hi[0] - lo[0]) * (i / 4.0)
                    x[1] = lo[1] + (hi[1] - lo[1]) * (j / 4.0)
                    x[2] = lo[2] + (hi[2] - lo[2]) * (k / 4.0)
                    sweep_points.append(x.copy())
        
        # Evaluate sweep points
        sweep_F = []
        for x in sweep_points:
            if used >= budget or is_time_up():
                break
            f = evaluate(x)
            sweep_F.append(f)
        
        # Phase 2: Local refinement around the best sweep points
        # Sort sweep points by hypervolume contribution (use simple sum as proxy)
        if len(sweep_F) > 0:
            sweep_F_arr = np.array(sweep_F)
            # Find non-dominated points in the sweep
            nd_indices = self._find_nondominated(sweep_F_arr)
            
            # Refine the best non-dominated points
            num_refine = min(50, budget - used)
            points_to_refine = [sweep_points[i] for i in nd_indices[:min(10, len(nd_indices))]]
            
            for pt in points_to_refine:
                for _ in range(num_refine // max(1, len(points_to_refine))):
                    if used >= budget or is_time_up():
                        break
                    # Gaussian perturbation
                    scale = 0.05 * range_
                    perturbed = pt + scale * rng.standard_normal(dim)
                    evaluate(perturbed)
        
        # Phase 3: Use remaining budget for additional random sampling with exploitation
        while used < budget and not is_time_up():
            # Mix of exploration and exploitation
            if rng.random() < 0.3:
                # Exploitation: perturb a good existing point
                if all_X:
                    idx = rng.integers(len(all_X))
                    scale = 0.03 * range_
                    cand = all_X[idx] + scale * rng.standard_normal(dim)
                    evaluate(cand)
                else:
                    evaluate(lo + range_ * rng.random(dim))
            else:
                # Exploration: random point
                evaluate(lo + range_ * rng.random(dim))
        
        # Return the last evaluated point (return value is ignored by evaluator)
        if all_X:
            return all_X[-1]
        return np.zeros(dim)

    def _find_nondominated(self, F):
        """Find indices of non-dominated points in F (n x m array)."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return np.where(nd)[0]
# EVOLVE-BLOCK-END
