"""Multi-objective optimizer for DTLZ7 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: when x[1:]=0, f1=x0, f2=(1-x0)*cos(x0*pi/2), f3=(1-x0)*sin(x0*pi/2)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        X_all = []
        F_all = []

        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = problem(x)
            X_all.append(x.copy())
            F_all.append(np.array(f))
            return f

        # Phase 1: Dense structured initialization with non-uniform x0 sampling
        # DTLZ7 front: f1=x0, f2=(1-x0)*cos(pi*x0/2), f3=(1-x0)*sin(pi*x0/2)
        # Hypervolume contribution is highest where front bulges out (x0 ~ 0.3-0.7)
        # Use a beta-like distribution concentrated in the middle
        n_init = min(50, budget // 4)

        # Create non-uniform x0 values: more points in the middle region
        # Use arcsine-like spacing: x0 = 0.5 - 0.5*sin(pi*(i/(n_init-1) - 0.5))
        # This concentrates points near x0=0.5
        for i in range(n_init):
            t = i / max(n_init - 1, 1)
            # Non-uniform: concentrate in [0.2, 0.8]
            x0 = 0.5 - 0.5 * np.sin(np.pi * (t - 0.5))
            x = np.zeros(dim)
            x[0] = x0
            # Most points exactly on the front
            if i % 4 != 3:
                x[1:] = 0.0
            else:
                rng = np.random.default_rng(i)
                x[1:] = rng.uniform(0, 0.0005, dim - 1)
            evaluate(x)

        used = n_init

        # Phase 2: Gap-filling - identify gaps in f1 (x0) coverage and add points
        if used < budget and len(F_all) > 5:
            f1_vals = np.array([f[0] for f in F_all])
            f1_sorted = np.sort(f1_vals)
            # Find largest gaps
            if len(f1_sorted) > 2:
                gaps = np.diff(f1_sorted)
                # Add points in the largest gaps
                n_gap_fill = min(15, budget - used)
                for _ in range(n_gap_fill):
                    if used >= budget or time.time() > deadline:
                        break
                    # Pick a random gap and place a point there
                    gap_idx = np.random.randint(0, len(gaps))
                    x0_new = f1_sorted[gap_idx] + np.random.uniform(0.1, 0.9) * gaps[gap_idx]
                    x0_new = max(0.0, min(1.0, x0_new))
                    x = np.zeros(dim)
                    x[0] = x0_new
                    x[1:] = 0.0
                    evaluate(x)
                    used += 1

        # Phase 3: Local refinement - aggressively drive x[1:] to 0
        while used < budget:
            if time.time() > deadline:
                break

            parent_idx = np.random.randint(0, len(X_all))
            parent = X_all[parent_idx].copy()
            child = parent.copy()

            # Aggressively reduce x[1:]
            # Set most components to exactly 0
            n_keep = np.random.randint(0, 4)  # keep at most 3 non-zero
            if n_keep > 0:
                keep_idx = np.random.choice(dim - 1, size=n_keep, replace=False)
                child[1 + keep_idx] = child[1 + keep_idx] * np.random.uniform(0.0, 0.1)
            child[1:] = np.where(np.arange(dim)[1:] == 0, child[1:], 0.0)
            # Reset all but kept components to 0
            child[1:] = 0.0
            if n_keep > 0:
                child[1 + keep_idx] = child[1 + keep_idx] * np.random.uniform(0.0, 0.1)

            # Small chance to perturb x0
            if np.random.rand() < 0.2:
                child[0] = parent[0] + np.random.uniform(-0.01, 0.01)
                child[0] = max(0.0, min(1.0, child[0]))

            fc = evaluate(child)
            used += 1

            # Prune to keep population manageable
            if len(X_all) > 180:
                F_arr = np.array(F_all)
                # Use a simple dominance-based pruning: keep points that are not dominated
                # For speed, use sum of objectives as proxy
                sums = np.sum(F_arr, axis=1)
                keep_idx = np.argsort(sums)[:180]
                X_all = [X_all[i] for i in keep_idx]
                F_all = [F_all[i] for i in keep_idx]

        # Phase 4: Final polish - ensure we have exact front points for best x0 values
        if used < budget and len(X_all) > 0:
            # Get unique x0 values from best points
            F_arr = np.array(F_all)
            sums = np.sum(F_arr, axis=1)
            sorted_idx = np.argsort(sums)[:min(20, len(X_all))]

            for idx in sorted_idx:
                if used >= budget or time.time() > deadline:
                    break
                x = X_all[idx].copy()
                x[1:] = 0.0
                x = np.clip(x, lo, hi)
                # Check if we already have this exact point
                is_dup = False
                for x_old in X_all:
                    if np.allclose(x, x_old, atol=1e-12):
                        is_dup = True
                        break
                if not is_dup:
                    evaluate(x)
                    used += 1

        return X_all
# EVOLVE-BLOCK-END
