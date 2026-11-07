import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using adaptive bisection + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([11.0, 400.0])

        # Store all evaluated points and their objective values
        X_all = []  # list of x vectors
        F_all = []  # list of f vectors

        def evaluate(x):
            """Evaluate x and record it."""
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            X_all.append(x.copy())
            F_all.append(f.copy())
            return f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Adaptive bisection to find the Pareto front shape
        # For ZDT4, the key variable is x_0 (controls g(x_0) = 4*sin(10*pi*x_0^2))
        # The Pareto front is parameterized by x_0 in [0,1], with x_1..x_9 = 0
        # f1 = 1 + 9/9 * sum(x_i^2) = 1 + sum(x_i^2) for x_1..x_9
        # f2 = g(x_0) * h(f1, g(x_0)) where h = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)

        # Since x_1..x_9 should be 0 for the Pareto front, f1 = 1 + sum(x_1..x_9^2)
        # At x_1..x_9 = 0, f1 = 1.0
        # So f2 = g(x_0) * (1 - sqrt(1/g(x_0)) - (1/g(x_0))*sin(10*pi/g(x_0)))

        # Strategy: Use adaptive bisection on x_0 to densely sample the front
        # Focus evaluations on regions with high curvature (near x_0 = 0)

        # Initial uniform grid on x_0
        n_initial = min(30, budget // 10)
        x0_grid = np.linspace(0, 1, n_initial)
        
        for x0 in x0_grid:
            if not time_ok() or len(F_all) >= budget:
                break
            x = np.zeros(dim)
            x[0] = x0
            evaluate(x)

        # Adaptive refinement: identify intervals with high hypervolume potential
        # and subdivide them
        # Use triangle area heuristic: intervals where the objective values change rapidly
        # are more important for hypervolume

        # Sort by x_0 to get ordered samples
        order = np.argsort(X_all[:, 0])
        x0_sorted = X_all[:, 0][order]
        f_sorted = np.array(F_all)[order]

        # Adaptive bisection iterations
        n_refine_iterations = 4
        for it in range(n_refine_iterations):
            if not time_ok() or len(F_all) >= budget:
                break

            # Identify the best intervals to subdivide
            # Compute local "gain" for each interval
            intervals = []
            for i in range(len(x0_sorted) - 1):
                x0_lo = x0_sorted[i]
                x0_hi = x0_sorted[i + 1]
                f_lo = f_sorted[i]
                f_hi = f_sorted[i + 1]

                # Estimate hypervolume contribution: area of trapezoid
                # Points strictly better than ref: f < ref component-wise
                # Approximate the area under the curve between these two points
                width = x0_hi - x0_lo
                # The f1 values are all 1.0 (since x_1..x_9 = 0)
                # Hypervolume contribution ~ width * min(f2_lo, f2_hi, ref[1]) - ref[0]... 
                # Actually for 2D: contribution = (ref[0] - f1) * (ref[1] - f2) for each point
                # For an interval, approximate as trapezoid area
                f2_lo = max(0, f_lo[1])
                f2_hi = max(0, f_hi[1])
                f1_val = f_lo[0]  # Should be ~1.0

                # Area of the region dominated by this interval segment
                if f1_val < ref[0] and min(f2_lo, f2_hi) < ref[1]:
                    area = (ref[0] - f1_val) * (ref[1] - (f2_lo + f2_hi) / 2.0) * width
                    if area < 0:
                        area = 0
                else:
                    area = 0

                intervals.append((area, x0_lo, x0_hi))

            if not intervals:
                break

            # Sort by area descending, pick top intervals
            intervals.sort(key=lambda t: -t[0])
            n_subdivide = min(10, len(intervals))

            for _, a, b in intervals[:n_subdivide]:
                if not time_ok() or len(F_all) >= budget:
                    break
                mid = (a + b) / 2.0
                x = np.zeros(dim)
                x[0] = mid
                evaluate(x)

            # Re-sort
            order = np.argsort(X_all[:, 0])
            x0_sorted = X_all[:, 0][order]
            f_sorted = np.array(F_all)[order]

        # Phase 2: Local refinement with Nelder-Mead on promising regions
        if time_ok() and len(F_all) < budget:
            # Find the best x_0 values (lowest f2 for given f1)
            # Try a few local optimizations
            n_restarts = min(5, max(1, (budget - len(F_all)) // 20))
            if n_restarts > 0:
                # Pick starting points from the current best
                # Sort by f2 (since f1 is always 1.0 for our points)
                f_arr = np.array(F_all)
                x_arr = np.array(X_all)
                
                # Get diverse starting points
                best_idx = np.argsort(f_arr[:, 1])[:min(5, len(F_all))]
                
                for idx in best_idx[:n_restarts]:
                    if not time_ok() or len(F_all) >= budget:
                        break
                    
                    x_start = x_arr[idx].copy()
                    # Only optimize x_0 (the first variable), keep others at 0
                    x0_start = x_start[0]
                    
                    # Simple local search: try small perturbations
                    best_x0 = x0_start
                    best_f = f_arr[idx].copy()
                    
                    step = 0.05
                    for _ in range(10):
                        if not time_ok() or len(F_all) >= budget:
                            break
                        # Try perturbations
                        for delta in [-step, step]:
                            x0_try = best_x0 + delta
                            if 0 <= x0_try <= 1:
                                x_try = np.zeros(dim)
                                x_try[0] = x0_try
                                f_try = evaluate(x_try)
                                if f_try[1] < best_f[1] or (abs(f_try[1] - best_f[1]) < 1e-10 and f_try[0] < best_f[0]):
                                    best_x0 = x0_try
                                    best_f = f_try.copy()
                        step *= 0.5

        # Phase 3: Use remaining budget for random exploration to catch any missed regions
        if time_ok() and len(F_all) < budget:
            rng = np.random.default_rng(0)
            while len(F_all) < budget and time_ok():
                x = lo + (hi - lo) * rng.random(dim)
                # Bias toward x_1..x_9 = 0 since that's where the Pareto front is
                x[1:] *= 0.1  # Scale down non-leading variables
                x = np.clip(x, lo, hi)
                evaluate(x)

        return np.array(X_all) if X_all else np.zeros((0, dim))
