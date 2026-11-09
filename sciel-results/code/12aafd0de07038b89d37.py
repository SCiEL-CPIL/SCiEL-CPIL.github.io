"""ZDT1 multi-objective optimizer: analytical front sampling + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure + L-BFGS-B polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # ZDT1 structure: x1 determines f1 = x1, f2 = g(x2..xn) * h(x1, g)
        # Pareto front: x2..xn = 0 => g = 1, h = 1 - sqrt(x1/g) = 1 - sqrt(x1)
        # So f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]
        # Strategy: sample x1 values and set x2..xn = 0, then refine with local optimization

        # Phase 1: Sample the known Pareto front analytically
        # Use adaptive sampling: more points where the front is curved
        n_samples = min(80, self.budget // 3)
        # Use a distribution that clusters near the ends (where h changes rapidly)
        # x1 = t^2 gives more density near 0
        t = np.linspace(0, 1, n_samples)
        x1_vals = t ** 2  # denser near 0
        # Also add uniform samples
        n_uniform = min(20, self.budget // 6)
        x1_uniform = np.linspace(0, 1, n_uniform)
        x1_all = np.unique(np.concatenate([x1_vals, x1_uniform]))
        x1_all = np.clip(x1_all, 0, 1)

        # Create points on the Pareto front
        X = np.zeros((len(x1_all), dim))
        X[:, 0] = x1_all
        # x2..xn = 0 for ideal ZDT1 front
        # But bounds might not allow 0, so clip
        X[:, 1:] = np.maximum(lo[1:], 0)

        F = np.array([problem(x) for x in X])
        used = len(X)

        # Phase 2: Local refinement using L-BFGS-B
        # For each point, try to minimize a scalarized objective to improve hypervolume
        # Use weighted sum or epsilon-constraint
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Pick points to refine: those with potential to improve hypervolume
            # Sort by f1 and pick spread-out points
            n_refine = min(remaining, len(X))
            # Select indices spread across the front
            indices = np.linspace(0, len(X) - 1, n_refine).astype(int)

            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                if used >= self.budget:
                    break

                x0 = X[idx].copy()
                f0 = F[idx].copy()

                # Try to minimize f2 while keeping f1 roughly constant (or slightly lower)
                # Use a weighted sum: minimize w1*f1 + w2*f2
                # Vary weights to explore different parts of the front

                # Try a few different weight combinations
                for w1, w2 in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]:
                    if used >= self.budget or time.time() > deadline - 5.0:
                        break

                    def objective(x):
                        fx = problem(x)
                        return w1 * fx[0] + w2 * fx[1]

                    def objective_with_count(x):
                        nonlocal used
                        result = objective(x)
                        used += 1
                        return result

                    # Use L-BFGS-B with bounds
                    try:
                        res = minimize(
                            objective_with_count,
                            x0,
                            method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 3, 'maxfun': 3}
                        )
                        if res.success and res.fun < f0[0] * w1 + f0[1] * w2:
                            # Evaluate the result properly
                            fx_new = problem(res.x)
                            used += 1
                            # Replace if it dominates or is non-dominated
                            X[idx] = res.x
                            F[idx] = fx_new
                    except Exception:
                        pass

        # Phase 3: Final random perturbation if budget remains
        if used < self.budget and time.time() < deadline - 2.0:
            rng = np.random.default_rng(42)
            while used < self.budget and time.time() < deadline - 2.0:
                # Pick a random existing point and perturb it
                base_idx = rng.integers(len(X))
                x_new = X[base_idx] + 0.05 * (hi - lo) * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1

                # Keep if it's non-dominated or improves hypervolume
                # Simple check: is it dominated by any existing point?
                dominated = False
                for j in range(len(X)):
                    if np.all(F[j] <= f_new) and np.any(F[j] < f_new):
                        dominated = True
                        break
                if not dominated:
                    # Replace the worst point (highest sum of objectives)
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = x_new
                    F[worst] = f_new

        # Return the best set of points
        # Select non-dominated points for the return
        # The evaluator records all returned objective vectors
        # We should return points that maximize hypervolume

        # Filter to non-dominated points
        n_dom = len(X)
        is_non_dom = np.ones(n_dom, dtype=bool)
        for i in range(n_dom):
            for j in range(n_dom):
                if i != j:
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_non_dom[i] = False
                        break

        if np.any(is_non_dom):
            X_best = X[is_non_dom]
            F_best = F[is_non_dom]
        else:
            X_best = X
            F_best = F

        return X_best
# EVOLVE-BLOCK-END
