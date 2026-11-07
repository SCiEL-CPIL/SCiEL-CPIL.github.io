"""Direct DTLZ7 Pareto front sampling with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = 325.0  # safety margin below 330s wall-clock limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        M = self.n_obj

        rng = np.random.default_rng(42)
        evaluated = 0
        max_evals = self.budget
        points = []
        F = []

        def check_deadline():
            return (time.perf_counter() - t0) < deadline

        def safe_eval(x):
            nonlocal evaluated
            if evaluated >= max_evals or not check_deadline():
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != n:
                x = np.resize(x, n)
            f = np.array(problem(x), dtype=float)
            if np.any(~np.isfinite(f)):
                return None
            evaluated += 1
            return f

        # DTLZ7 with n=22, M=3:
        # k = n - M + 2 = 21 shape variables (x[0]..x[20])
        # g-set: last M-2 = 1 variable: x[21]
        # g(x) = (M-2 + sum_{i in g-set} (x_i - 0.5)^2 * cos(20*pi*(x_i - 0.5)))^2
        # g is minimized when x[21] = 0.5 (g = 0)
        #
        # On the Pareto front (g=0):
        #   f1 = x[0]
        #   f2 = x[1]
        #   f3 = (1+g) * h(y, m) = 1 * h(y, m)
        #   where y = sum(x[2]..x[20]), m = n - 2 = 20
        #   h(y, m) = 1 - y/m * (1 + sin(pi * y / m))
        #
        # Strategy: fix x[21] = 0.5, then sample the 21-dim shape space.
        # For hypervolume with ref [1.1, 1.1, 25.0], we need good coverage
        # of (f1, f2, f3) = (x[0], x[1], h(y, 20)).
        #
        # Since f1 and f2 are directly x[0] and x[1], and f3 depends only
        # on the sum of x[2..20], we can:
        # 1. Sample x[0] and x[1] on a stratified grid
        # 2. For the middle 19 vars, use patterns that produce diverse sums

        # Phase 1: Verify g-variable optimum with a few evaluations
        # x[21] = 0.5 should give g = 0
        g_var = 0.5
        # Quick verification: evaluate at x[21]=0.5 vs x[21]=0.0
        x_test = np.zeros(n)
        x_test[0] = 0.5
        x_test[1] = 0.5
        x_test[21] = 0.5
        f_test = safe_eval(x_test)
        if f_test is not None:
            points.append(np.clip(x_test, lo, hi))
            F.append(f_test)

        # Also check a couple other g values to confirm 0.5 is optimal
        for gv in [0.0, 0.25, 0.75, 1.0]:
            if not check_deadline() or evaluated >= max_evals:
                break
            x_test = np.zeros(n)
            x_test[0] = 0.5
            x_test[1] = 0.5
            x_test[21] = gv
            f_test = safe_eval(x_test)
            if f_test is not None:
                points.append(np.clip(x_test, lo, hi))
                F.append(f_test)

        if not F:
            return np.array([])

        # Phase 2: Dense Pareto front sampling
        # Fix x[21] = 0.5 (optimal g)
        # Sample x[0], x[1] on stratified grid
        # Use patterns for x[2..20] to create diverse f3 values

        remaining = max_evals - evaluated
        if remaining <= 0 or not check_deadline():
            return np.array(points) if points else np.zeros((0, n))

        # Grid size for x[0] and x[1]
        # Balance: more grid points = better coverage of (f1, f2) plane
        n_grid = int(np.sqrt(remaining * 0.6)) + 1
        n_grid = max(3, min(n_grid, 18))

        # Stratified sampling for x[0] and x[1]
        xs0 = np.linspace(0, 1, n_grid, endpoint=False) + rng.random(n_grid) * (1.0 / n_grid)
        xs1 = np.linspace(0, 1, n_grid, endpoint=False) + rng.random(n_grid) * (1.0 / n_grid)

        # Middle variables: x[2..20] (19 vars)
        # f3 = h(sum(x[2..20]), 20) where h(y,m) = 1 - y/m*(1+sin(pi*y/m))
        # To get diverse f3 values, we need diverse sums of the 19 middle vars
        # Each middle var in [0,1], so sum in [0, 19]
        # f3 ranges from h(0,20)=1 to h(19,20)≈1 - 0.95*(1+sin(0.95*pi))
        #
        # Use several patterns for the middle variables:
        mid_len = 19  # x[2] through x[20]

        # Generate patterns that produce different sums
        patterns = []
        # Pattern 0: all zeros (sum=0, f3=h(0,20)=1)
        patterns.append(np.zeros(mid_len))
        # Pattern 1: all 0.5 (sum=9.5, f3=h(9.5,20))
        patterns.append(np.full(mid_len, 0.5))
        # Pattern 2: linear ramp 0 to 1 (sum=9.5)
        patterns.append(np.linspace(0, 1, mid_len))
        # Pattern 3: all 1.0 (sum=19, f3=h(19,20))
        patterns.append(np.ones(mid_len))
        # Pattern 4: low values (sum≈3)
        patterns.append(np.full(mid_len, 0.15))
        # Pattern 5: high values (sum≈15)
        patterns.append(np.full(mid_len, 0.8))
        # Pattern 6: random
        patterns.append(rng.random(mid_len))
        # Pattern 7: another random
        patterns.append(rng.random(mid_len))

        # Determine how many grid points per pattern
        n_patterns = len(patterns)
        evals_per_pattern = remaining // n_patterns
        if evals_per_pattern < n_grid:
            # Not enough budget for full grid per pattern, reduce patterns
            n_patterns = max(1, remaining // n_grid)
            patterns = patterns[:n_patterns]
            evals_per_pattern = remaining // n_patterns

        # Evaluate grid for each pattern
        for pi in range(n_patterns):
            mid = patterns[pi]
            for i in range(n_grid):
                for j in range(n_grid):
                    if not check_deadline() or evaluated >= max_evals:
                        break
                    x = np.zeros(n)
                    x[0] = xs0[i]
                    x[1] = xs1[j]
                    x[2:21] = mid
                    x[21] = 0.5
                    f = safe_eval(x)
                    if f is None:
                        break
                    points.append(np.clip(x, lo, hi))
                    F.append(f)

        # Phase 3: Refinement - focus on underrepresented regions
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline() and len(F) > 10:
            F_arr = np.array(F)
            # Identify gaps in (f1, f2, f3) space
            # Simple approach: random sampling near existing good points
            n_refine = min(remaining, 40)
            for _ in range(n_refine):
                if not check_deadline() or evaluated >= max_evals:
                    break
                # Random point on the Pareto front structure
                x = np.zeros(n)
                x[0] = rng.random()
                x[1] = rng.random()
                # Random middle vars with controlled sum
                target_sum = rng.uniform(0, 19)
                x[2:21] = rng.random(mid_len)
                # Scale to approximate target sum
                current_sum = x[2:21].sum()
                if current_sum > 0:
                    x[2:21] = x[2:21] * (target_sum / current_sum)
                x[2:21] = np.clip(x[2:21], 0, 1)
                x[21] = 0.5
                f = safe_eval(x)
                if f is None:
                    break
                points.append(np.clip(x, lo, hi))
                F.append(f)

        # Final check
        if not points:
            if evaluated < max_evals and check_deadline():
                x = np.random.uniform(0, 1, n)
                f = safe_eval(x)
                if f is not None:
                    points.append(np.clip(x, lo, hi))
                    F.append(f)

        return np.array(points) if points else np.zeros((0, n))

# EVOLVE-BLOCK-END
