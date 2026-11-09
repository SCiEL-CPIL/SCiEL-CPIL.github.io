"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pareto front sampling with local refinement for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Strategy: ZDT2 Pareto front has f1 = x1, and for optimal solutions
        # x2..xn are set to achieve the front. We sample x1 values and
        # optimize x2..xn to minimize f2 for each x1.
        # Also do random search as backup.

        n_samples = min(150, budget // 2)
        # Sample x1 values - focus on region where f2 might be small
        # Try a range of x1 values
        x1_values = np.linspace(0.01, 1.0, n_samples)

        # For each x1, we need to find x2..xn that minimizes f2
        # In ZDT2, f2 = (1+10n) * (1 - sqrt(f1/(1+10n))) when x2..xn are set optimally
        # But we don't know the exact form, so we use local optimization

        used = 0
        all_x = []
        all_f = []

        # Phase 1: Sample along x1 axis with optimized x2..xn
        # Use a simple approach: for each x1, start with x2..xn = 0.5 and do a few gradient-free steps
        for x1 in x1_values:
            if used >= budget or time.time() > deadline:
                break

            # Initial point: x1 fixed, x2..xn random
            x = np.full(dim, 0.5)
            x[0] = x1
            x = np.clip(x, lo, hi)

            # Evaluate
            f = np.array(problem(x), dtype=float)
            used += 1
            all_x.append(x.copy())
            all_f.append(f.copy())

            # Try to improve f2 by adjusting x2..xn
            # Use a few random perturbations
            for step in range(2):
                if used >= budget or time.time() > deadline:
                    break
                # Perturb x2..xn
                x_trial = x.copy()
                x_trial[1:] += 0.3 * (hi[1:] - lo[1:]) * np.random.randn(dim - 1)
                x_trial = np.clip(x_trial, lo, hi)
                f_trial = np.array(problem(x_trial), dtype=float)
                used += 1

                # Keep if better in at least one objective without worsening the other too much
                if f_trial[0] <= f[0] + 0.01 and f_trial[1] < f[1]:
                    x = x_trial
                    f = f_trial
                    all_x.append(x.copy())
                    all_f.append(f.copy())

        # Phase 2: Random search with remaining budget
        if used < budget:
            remaining = budget - used
            n_random = min(remaining, 50)
            for _ in range(n_random):
                if used >= budget or time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(dim)
                f = np.array(problem(x), dtype=float)
                used += 1
                all_x.append(x.copy())
                all_f.append(f.copy())

        # Phase 3: Local refinement of best points
        if used < budget and len(all_f) > 0:
            all_f = np.array(all_f)
            all_x = np.array(all_x)

            # Find non-dominated points
            n_pts = len(all_f)
            non_dom = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                if not non_dom[i]:
                    continue
                for j in range(n_pts):
                    if i == j or not non_dom[j]:
                        continue
                    if np.all(all_f[j] <= all_f[i]) and np.any(all_f[j] < all_f[i]):
                        non_dom[i] = False
                        break

            nd_idx = np.where(non_dom)[0]
            if len(nd_idx) > 0:
                # Refine the best non-dominated points
                n_refine = min(20, budget - used)
                for idx in nd_idx[:n_refine]:
                    if used >= budget or time.time() > deadline:
                        break
                    x_best = all_x[idx].copy()
                    f_best = all_f[idx].copy()

                    # Local search with decreasing step size
                    for step_size in [0.1, 0.05, 0.02, 0.01]:
                        for _ in range(3):
                            if used >= budget or time.time() > deadline:
                                break
                            x_trial = x_best.copy()
                            x_trial += step_size * (hi - lo) * np.random.randn(dim)
                            x_trial = np.clip(x_trial, lo, hi)
                            f_trial = np.array(problem(x_trial), dtype=float)
                            used += 1

                            if np.all(f_trial <= f_best) and np.any(f_trial < f_best):
                                x_best = x_trial
                                f_best = f_trial
                                all_x.append(x_best.copy())
                                all_f.append(f_best.copy())

        return all_x
# EVOLVE-BLOCK-END
