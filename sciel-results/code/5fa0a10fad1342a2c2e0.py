"""Combined optimizer for BBOB f24 (Lunacek bi-Rastrigin):
coarse structured basin scan + (1+1)-ES with 1/5 success rule and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse scan to locate basins, then (1+1)-ES (1/5 rule) with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # safety margin before the 300s limit

        def safe_func(x):
            """Wrap func to handle potential edge cases."""
            try:
                val = func(x)
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        def record(x, f):
            nonlocal best_x, best_f, call_count
            if f < best_f:
                best_x = x.copy()
                best_f = f
            call_count += 1

        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            f = safe_func(x)
            record(x, f)
            return f

        # Phase 1: Coarse structured scan to locate basins
        # Lunacek bi-Rastrigin has two basins; try to find both
        # Use a grid-like scan with some randomness
        n_scan = min(40, budget // 10)
        
        # Try center and corners of the box
        scan_points = []
        # Center
        scan_points.append(np.zeros(dim))
        # Corners of [-5,5]^dim
        for i in range(2 ** min(dim, 4)):
            pt = np.array([(i >> j) & 1 for j in range(min(dim, 4))], dtype=float)
            pt = pt * 10 - 5
            if len(pt) < dim:
                pt = np.concatenate([pt, np.zeros(dim - len(pt))])
            scan_points.append(pt)
        
        # Add some random points
        for _ in range(n_scan - len(scan_points)):
            scan_points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate scan points
        for pt in scan_points[:n_scan]:
            if call_count >= budget or time.time() > deadline:
                break
            eval_pt(pt)

        # Phase 2: Run (1+1)-ES from the best found point
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
        else:
            x = rng.uniform(-5, 5, dim)
            fx = eval_pt(x)

        sigma = 1.0
        sigma_min = 1e-8
        sigma_max = 10.0

        # (1+1)-ES with 1/5 success rule
        successes = 0
        total = 0
        restart_count = 0
        max_restarts = 3

        while call_count < budget and time.time() < deadline:
            # Check for restart condition
            if total >= 20:
                success_rate = successes / total
                if success_rate < 0.05 or success_rate > 0.45:
                    # Consider restart
                    if restart_count < max_restarts:
                        restart_count += 1
                        # Restart from best point with smaller sigma
                        x = best_x.copy() if best_x is not None else x
                        fx = best_f
                        sigma = 0.5
                        successes = 0
                        total = 0
                        continue

            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = eval_pt(y)
            
            total += 1
            if fy < fx:
                x, fx = y, fy
                successes += 1
                sigma *= 1.5
            else:
                sigma *= 0.87

            sigma = max(sigma_min, min(sigma, sigma_max))

        # Phase 3: If we have remaining budget, do local refinement
        if call_count < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            # Local refinement with smaller steps
            for _ in range(min(50, budget - call_count)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_pt(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                sigma = max(1e-8, min(sigma, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
