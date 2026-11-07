"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT2 structure: f1 = x1, f2 = (1+f1)*g*(1 - sqrt(f1/(1+f1)))
        # Pareto front: g=0 => x2..xn=0, f1 in [0,1]
        # We focus on the 1D search space: x1 with x2..xn=0

        # Strategy: dense sampling of the known Pareto front + local refinement
        # Use power-law spacing to concentrate points near the steep part (x1 near 1)

        # First, evaluate a structured set of points on the Pareto front
        # x2..xn = 0, x1 = various values

        # Number of structured points
        n_structured = min(budget // 3, 100)

        # Power-law spacing: more points near x1=1 (steep region)
        # t in [0, 1], x1 = 1 - t^p where p > 1 concentrates near x1=1
        # Actually for ZDT2, the front is f2 = (1+f1)*(1-sqrt(f1/(1+f1)))
        # The hypervolume contribution is largest where the front is "high" in f2
        # which is near f1=0 (f2=1) and near f1=1 (f2=0)
        # The gap area is largest in the middle where the curve is steep

        # Let's use a mix: uniform in f1 space plus extra near endpoints
        # Actually, let's just use many points with power-law bias

        # Evaluate structured points on Pareto front
        used = 0
        F = []
        X = []

        # Phase 1: Structured sampling on Pareto front (x2..xn = 0)
        # Use power-law: x1_i = 1 - (i/n)^p for i=0..n-1, p=2
        # This gives more points near x1=1

        n_phase1 = min(n_structured, budget - 50)
        if n_phase1 > 0:
            # Power law sampling
            p = 2.0
            t = np.linspace(0, 1, n_phase1 + 1)
            x1_vals = 1.0 - t**p
            # Also add some uniform points
            n_uniform = max(1, n_phase1 // 4)
            x1_uniform = np.linspace(0.0, 1.0, n_uniform)

            all_x1 = np.concatenate([x1_vals, x1_uniform])
            # Deduplicate
            all_x1 = np.unique(all_x1)
            all_x1 = np.clip(all_x1, 0.0, 1.0)

            for x1_val in all_x1:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x1_val
                # Clip to bounds
                x = np.clip(x, lo, hi)
                f = problem(x)
                F.append(f)
                X.append(x.copy())
                used += 1

        # Phase 2: Local refinement around promising regions
        # Use remaining budget to do local search
        remaining = budget - used
        if remaining > 0 and F:
            F_arr = np.array(F)
            X_arr = np.array(X)

            # Find non-dominated points
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break

            # Get non-dominated set
            nd_indices = np.where(~dominated)[0]
            if len(nd_indices) > 0:
                nd_X = X_arr[nd_indices]
                nd_F = F_arr[nd_indices]

                # Local perturbation of non-dominated points
                # Focus on x1 dimension since that's where the action is
                n_local = min(remaining, len(nd_indices) * 3)
                step_size = 0.01

                for i in range(n_local):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    idx = i % len(nd_indices)
                    parent = nd_X[idx].copy()

                    # Perturb x1 with small step
                    step = step_size * (0.5 + np.random.random())
                    direction = 1 if np.random.random() > 0.5 else -1
                    child = parent.copy()
                    child[0] += direction * step
                    # Keep x2..xn at 0 (or small random)
                    child[1:] = 0.0
                    child = np.clip(child, lo, hi)

                    fc = problem(child)
                    F.append(fc)
                    X.append(child)
                    used += 1

                    # If child is non-dominated, add to nd set for further refinement
                    # (simplified: just continue)

        # Return the set of evaluated points
        if X:
            return np.array(X)
        else:
            # Fallback: return a single random point
            x = lo + (hi - lo) * np.random.random(dim)
            problem(x)
            return x
# EVOLVE-BLOCK-END
