"""Hybrid LHS + BFGS optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start with stratified LHS grid and BFGS local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            try:
                fx = func(x)
                calls_made += 1
                if not np.isfinite(fx):
                    return np.inf
                return float(fx)
            except Exception:
                calls_made += 1
                return np.inf

        def check_time():
            return time.time() < deadline

        # Phase 1: Stratified LHS sampling for global coverage
        # Use a reasonable number of LHS points, limited by budget
        n_lhs = min(50, budget // 10)
        if n_lhs < 5:
            n_lhs = 5

        rng = np.random.default_rng(42)
        lhs_points = self._generate_lhs(n_lhs, dim, rng)

        # Evaluate LHS points
        for x in lhs_points:
            if not check_time():
                break
            if calls_made >= budget:
                break
            x_clipped = np.clip(x, -5, 5)
            fx = safe_call(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()

        # Phase 2: BFGS local search from each promising starting point
        # Sort starting points by function value and run BFGS from top candidates
        # We'll collect all evaluated points and sort them
        evaluated_points = []
        for x in lhs_points:
            x_clipped = np.clip(x, -5, 5)
            # Re-evaluate if needed (but we already have values)
            # Actually let's just track them
            pass

        # Let's redo this more carefully - track all points and their values
        # Reset and do it properly
        # Actually, let me restructure

        # Phase 2: Run BFGS from the best few starting points
        # First, let's gather candidate starting points
        # We'll use the LHS points sorted by their function values
        # But we need to re-evaluate or store them

        # Let me restructure: store all evaluations
        all_points = []
        all_values = []

        for x in lhs_points:
            if not check_time():
                break
            if calls_made >= budget:
                break
            x_clipped = np.clip(x, -5, 5)
            fx = safe_call(x_clipped)
            all_points.append(x_clipped)
            all_values.append(fx)

        # Sort by value
        if all_values:
            sorted_idx = np.argsort(all_values)
            # Try BFGS from top candidates
            n_bfgs_starts = min(5, len(sorted_idx))
            for i in range(n_bfgs_starts):
                if not check_time():
                    break
                if calls_made >= budget:
                    break

                x0 = all_points[sorted_idx[i]].copy()

                # Run BFGS with limited iterations
                result = self._run_bfgs(x0, func, safe_call, check_time, budget, calls_made)
                if result is not None:
                    x_bfgs, f_bfgs = result
                    if f_bfgs < best_f:
                        best_f = f_bfgs
                        best_x = x_bfgs.copy()

        # Phase 3: If we have budget left, do additional random restarts with BFGS
        remaining = budget - calls_made
        if remaining > 10 and check_time():
            n_extra = min(20, remaining // 5)
            for _ in range(n_extra):
                if not check_time():
                    break
                if calls_made >= budget:
                    break

                x0 = rng.uniform(-5, 5, dim)
                x0 = np.clip(x0, -5, 5)
                fx0 = safe_call(x0)
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()

                # Run BFGS from this point
                result = self._run_bfgs(x0, func, safe_call, check_time, budget, calls_made)
                if result is not None:
                    x_bfgs, f_bfgs = result
                    if f_bfgs < best_f:
                        best_f = f_bfgs
                        best_x = x_bfgs.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        return best_x, best_f

    def _generate_lhs(self, n, dim, rng):
        """Generate Latin Hypercube Sampling points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            # Create stratified intervals
            edges = np.linspace(-5, 5, n + 1)
            for i in range(n):
                # Random point within each stratum
                points[i, d] = rng.uniform(edges[i], edges[i + 1])
        # Shuffle rows to decorrelate
        perm = rng.permutation(n)
        points = points[perm]
        return points

    def _run_bfgs(self, x0, func, safe_call, check_time, budget, calls_made):
        """Run BFGS local search with a wrapper that respects budget and time."""
        dim = len(x0)

        # Wrap the function for scipy.optimize.minimize
        # We need to track calls within this BFGS run
        local_calls = [0]

        def wrapped_func(x):
            if not check_time():
                return np.inf
            if calls_made + local_calls[0] >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = safe_call(x_clipped)
            local_calls[0] += 1
            return fx

        try:
            result = minimize(
                wrapped_func,
                x0,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-8, 'maxcor': 10}
            )
            if result.success or result.fun < np.inf:
                x_opt = np.clip(result.x, -5, 5)
                f_opt = result.fun
                if np.isfinite(f_opt):
                    return x_opt, f_opt
        except Exception:
            pass

        return None
# EVOLVE-BLOCK-END
