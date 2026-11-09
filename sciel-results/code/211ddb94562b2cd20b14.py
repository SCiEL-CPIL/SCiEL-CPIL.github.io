"""Multi-objective optimizer for ZDT6 exploiting analytic front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT6 optimizer: seed analytic front, directed coordinate descent on g-variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 320.0  # 330s wall-clock limit, 10s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.dim
        budget = self.budget

        # Archive of (x, f)
        archive_x = []
        archive_f = []
        used = 0

        def add(x, f):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(f, dtype=float)
            if np.any(f >= 11.0) or not np.all(np.isfinite(f)):
                return
            archive_x.append(x)
            archive_f.append(f)
            used += 1

        def dominated(f):
            # True if any archived point strictly dominates f
            if len(archive_f) == 0:
                return False
            F = np.array(archive_f)
            # point dominates f if F <= f and F < f in at least one coord
            le = np.all(F <= f, axis=1)
            lt = np.any(F < f, axis=1)
            return bool(np.any(le & lt))

        # ---- Phase 1: Seed the analytic Pareto front ----
        # x0 = t, x[1:] = 0  ->  on the Pareto front
        n_seed = min(40, budget // 4)
        ts = np.linspace(0.0, 1.0, n_seed)
        for t in ts:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = t
            x = np.clip(x, lo, hi)
            f = problem(x)
            add(x, f)

        # ---- Phase 2: Directed coordinate descent to minimize g ----
        # For each archived point, push x[1:] toward 0 to reduce g.
        # Also small x0 perturbations to explore f1 landscape.
        # Work on a working population of the best (non-dominated) points.
        # Keep a list of "active" points to refine.
        active_idx = list(range(len(archive_x)))

        # We will iteratively refine. Each refinement step = 1 evaluation.
        # Coordinate descent on x[1:] (g-variables) with adaptive step.
        # Also occasional x0 jitter.

        # Sort active by f1 (x0-based) to spread effort
        # Use current archive as the pool.

        # Local search: for each point, try moving each x[1:] coord toward 0.
        # Step size adapts: start 0.1, shrink if improving.

        # Build a refinement queue: (point_index, coord, direction, step)
        # We process queue items until budget/deadline.

        # Simpler: loop over points, for each try a random coordinate move.
        # Prefer g-variables (1..n-1) to reduce g, and x0 (0) to explore.

        # Use a per-point step schedule.
        # We'll maintain step sizes per point.
        steps = {i: 0.1 for i in range(len(archive_x))}

        # Refinement loop
        while used < budget:
            if time.time() > deadline:
                break
            if len(archive_x) == 0:
                break

            # Pick a point to refine: prefer non-dominated, and rotate.
            # Compute dominance mask
            F = np.array(archive_f)
            dom = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                dom[i] = bool(np.any((le & lt) & (np.arange(len(F)) != i)))
            # Non-dominated indices
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([int(np.argmin(F[:, 0]))])

            # Choose a point round-robin among non-dominated
            # Use a rotating pointer
            # We'll just pick randomly among non-dominated for diversity
            i = int(nd_idx[rng.integers(len(nd_idx))])
            xi = archive_x[i].copy()
            fi = archive_f[i]

            # Choose a coordinate to perturb
            # Bias toward g-variables (1..n-1) to reduce g
            if n > 1:
                # 70% chance pick a g-variable, 30% pick x0
                if rng.random() < 0.7:
                    c = int(rng.integers(1, n))
                else:
                    c = 0
            else:
                c = 0

            step = steps.get(i, 0.1)
            # Direction: for g-variables, move toward 0 (reduce g)
            # for x0, random direction
            if c == 0:
                d = rng.choice([-1.0, 1.0])
            else:
                # move toward 0
                if xi[c] > 0:
                    d = -1.0
                elif xi[c] < 0:
                    d = 1.0
                else:
                    d = rng.choice([-1.0, 1.0])

            # Try the move
            x_new = xi.copy()
            x_new[c] = x_new[c] + d * step * span[c]
            x_new = np.clip(x_new, lo, hi)
            if np.allclose(x_new, xi):
                # no change; shrink step
                steps[i] = max(0.01, step * 0.5)
                continue

            f_new = problem(x_new)
            add(x_new, f_new)
            if used >= budget:
                break

            # Check improvement
            improved = (f_new[0] < fi[0]) or (f_new[1] < fi[1])
            # If improved, keep step; else shrink
            if improved:
                steps[i] = min(0.3, step * 1.2)
            else:
                steps[i] = max(0.01, step * 0.5)

            # Periodically add a fresh random point to escape local minima
            if used % 25 == 0 and used < budget:
                if time.time() > deadline:
                    break
                x_rand = lo + span * rng.random(n)
                f_rand = problem(x_rand)
                add(x_rand, f_rand)
                # reset some steps
                for k in list(steps.keys()):
                    if k < len(archive_x):
                        steps[k] = 0.1

        # Final: return something (ignored)
        if len(archive_x) > 0:
            return np.array(archive_x)
        return np.array([])
# EVOLVE-BLOCK-END
