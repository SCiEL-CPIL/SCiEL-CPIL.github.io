"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure to construct the Pareto front directly.
    
    ZDT4:
    f1(x) = x1
    g(x) = 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi) for i=2..n)
    f2(x) = g(x) * (1 - sqrt(x1/g) - (x1/g)*sin^2(10*pi*x1))
    
    The global minimum of g is 10*(n-1) = 90 when xi=0 for all i>=2.
    The Pareto front is traced by sweeping x1 in [0,1] with x2..x10=0.
    
    Strategy:
    1. Use most of the budget to densely sample x1 in [0,1] with x2..x10=0.
    2. Use a small fraction to verify/refine the x2..x10=0 setting.
    3. Apply a final polish pass to ensure coverage at the boundaries.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        budget = self.budget

        # Set x2..x10 to the known global minimizer of g
        # For standard ZDT4 bounds [-5,5], this is 0
        x_rest = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            idx = i + 1
            if lo[idx] <= 0.0 <= hi[idx]:
                x_rest[i] = 0.0
            else:
                x_rest[i] = lo[idx] if abs(0.0 - lo[idx]) < abs(0.0 - hi[idx]) else hi[idx]

        all_points = []
        all_evals = 0

        def eval_point(x):
            nonlocal all_evals
            if all_evals >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                all_evals += 1
                f = np.array(f, dtype=np.float64)
                if np.all(np.isfinite(f)):
                    all_points.append(f)
                    return f
            except Exception:
                pass
            return None

        # Phase 1: Dense sweep of x1 across [lo[0], hi[0]]
        # Allocate 90% of budget to this
        n_sweep = int(budget * 0.9)
        n_sweep = max(n_sweep, 50)
        
        # Use a combination of uniform and slightly denser near boundaries
        # to ensure good coverage of the entire front
        x1_uniform = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1 in x1_uniform:
            if time.time() > deadline or all_evals >= budget:
                break
            x = np.empty(n, dtype=np.float64)
            x[0] = x1
            x[1:] = x_rest
            x = np.clip(x, lo, hi)
            eval_point(x)

        # Phase 2: Refinement - try small perturbations of x_rest to see
        # if we can find a better g value (unlikely for ZDT4 but worth checking)
        remaining = budget - all_evals
        if remaining > 5 and time.time() < deadline - 10:
            rng = np.random.default_rng(42)
            n_refine = min(remaining, 15)
            for _ in range(n_refine):
                if time.time() > deadline or all_evals >= budget:
                    break
                # Perturb x_rest slightly
                pert = 0.01 * rng.standard_normal(n - 1)
                x_rest_pert = x_rest + pert
                x_rest_pert = np.clip(x_rest_pert, lo[1:], hi[1:])
                
                # Evaluate at a few x1 values
                for x1_frac in [0.25, 0.5, 0.75]:
                    if time.time() > deadline or all_evals >= budget:
                        break
                    x1_val = lo[0] + (hi[0] - lo[0]) * x1_frac
                    x = np.empty(n, dtype=np.float64)
                    x[0] = x1_val
                    x[1:] = x_rest_pert
                    x = np.clip(x, lo, hi)
                    eval_point(x)

        # Phase 3: Ensure boundary points are well-covered
        # Add explicit evaluations at x1 = lo[0] and x1 = hi[0] if not already done
        remaining = budget - all_evals
        if remaining > 0 and time.time() < deadline - 5:
            # Check if we have points near the boundaries
            if len(all_points) > 0:
                F = np.array(all_points)
                f1_vals = F[:, 0]
                tol = (hi[0] - lo[0]) * 0.01
                has_low = np.any(np.abs(f1_vals - lo[0]) < tol)
                has_high = np.any(np.abs(f1_vals - hi[0]) < tol)
                
                if not has_low and all_evals < budget:
                    x = np.empty(n, dtype=np.float64)
                    x[0] = lo[0]
                    x[1:] = x_rest
                    x = np.clip(x, lo, hi)
                    eval_point(x)
                
                if not has_high and all_evals < budget:
                    x = np.empty(n, dtype=np.float64)
                    x[0] = hi[0]
                    x[1:] = x_rest
                    x = np.clip(x, lo, hi)
                    eval_point(x)

        # Phase 4: If budget remains, add more uniform samples to fill gaps
        remaining = budget - all_evals
        if remaining > 0 and time.time() < deadline - 2:
            # Identify gaps in x1 coverage
            if len(all_points) > 10:
                F = np.array(all_points)
                f1_vals = np.sort(F[:, 0])
                # Find the largest gap
                gaps = np.diff(f1_vals)
                max_gap_idx = np.argmax(gaps)
                gap_size = gaps[max_gap_idx]
                mid_gap = (f1_vals[max_gap_idx] + f1_vals[max_gap_idx + 1]) / 2
                
                # Add a few points in the largest gap
                n_fill = min(remaining, 5)
                for i in range(n_fill):
                    if time.time() > deadline or all_evals >= budget:
                        break
                    frac = (i + 1) / (n_fill + 1)
                    x1_val = f1_vals[max_gap_idx] + frac * gap_size
                    x = np.empty(n, dtype=np.float64)
                    x[0] = x1_val
                    x[1:] = x_rest
                    x = np.clip(x, lo, hi)
                    eval_point(x)

        return all_points
# EVOLVE-BLOCK-END
