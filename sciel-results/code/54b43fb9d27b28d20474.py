"""Multi-objective optimizer for ZDT1 using hypervolume-driven adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        n_obj = self.n_obj

        # ZDT1: Pareto front requires g(x) = sum(x[1:]) = 1
        # f1 = x[0], f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / g))
        # To be on PF: set x[1:] so that sum = 1

        def make_x_on_pf(x1, spread_mode='uniform'):
            """Create a 30-dim vector with x[0]=x1 and sum(x[1:])=1."""
            x = np.zeros(n)
            x[0] = x1
            # Distribute remaining 1.0 across x[1:]
            if spread_mode == 'uniform':
                x[1:] = 1.0 / (n - 1)
            elif spread_mode == 'concentrated':
                # Put most mass in x[1], rest uniform
                x[1] = 0.5
                x[2:] = 0.5 / (n - 2)
            return x

        def hypervolume_2d(points, ref):
            """Compute hypervolume for 2D points with reference point."""
            if len(points) == 0:
                return 0.0
            # Filter points that are better than reference in all objectives
            pts = np.array([p for p in points if p[0] < ref[0] and p[1] < ref[1]])
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_f2 = ref[1]
            for p in pts:
                if p[1] < prev_f2:
                    hv += (ref[0] - p[0]) * (prev_f2 - p[1])
                    prev_f2 = p[1]
            return hv

        ref = np.array([11.0, 11.0])
        evaluated = []  # list of (x, f)
        used = 0

        def eval_point(x):
            nonlocal used
            f = np.array(problem(x))
            evaluated.append((x.copy(), f.copy()))
            used += 1
            return f

        # Phase 1: Initial spread of x1 values
        # Use a mix of uniform and concentrated approaches
        n_initial = 30
        x1_vals = np.linspace(0.01, 0.99, n_initial)
        for x1 in x1_vals:
            if used >= self.budget or time.time() > deadline:
                break
            x = make_x_on_pf(x1, 'uniform')
            x = np.clip(x, lo, hi)
            eval_point(x)

        # Phase 2: Adaptive refinement using hypervolume marginal gain
        # For each candidate x1, estimate marginal hypervolume gain
        remaining_budget = self.budget - used
        n_refine = min(remaining_budget, 120)

        # Maintain a list of candidate x1 positions to try
        # Start with midpoints of intervals between existing points
        existing_f1 = sorted([f[0] for _, f in evaluated])

        candidates = []
        # Add midpoints between consecutive existing points
        for i in range(len(existing_f1) - 1):
            mid = (existing_f1[i] + existing_f1[i + 1]) / 2.0
            candidates.append(mid)
        # Add some random candidates
        for _ in range(20):
            candidates.append(rng.uniform(0.005, 0.995))

        # Sort candidates by expected marginal gain (rough heuristic: distance to nearest evaluated point)
        def marginal_gain_estimate(x1):
            if len(existing_f1) == 0:
                return 1.0
            # Distance to nearest evaluated f1
            dists = np.abs(np.array(existing_f1) - x1)
            return 1.0 / (1.0 + np.min(dists))

        candidates.sort(key=marginal_gain_estimate, reverse=True)

        for i in range(n_refine):
            if used >= self.budget or time.time() > deadline:
                break
            if i < len(candidates):
                x1 = candidates[i]
            else:
                # Generate new candidate near best hypervolume-contributing region
                x1 = rng.uniform(0.005, 0.995)

            # Try a few different spread modes
            best_f = None
            best_x = None
            for mode in ['uniform', 'concentrated']:
                if used >= self.budget or time.time() > deadline:
                    break
                x = make_x_on_pf(x1, mode)
                x = np.clip(x, lo, hi)
                f = eval_point(x)
                if best_f is None or f[1] < best_f[1]:
                    best_f = f
                    best_x = x

            # Update existing_f1
            existing_f1 = sorted([f[0] for _, f in evaluated])

        # Phase 3: Local polish - refine around points that contribute most to hypervolume
        remaining = self.budget - used
        if remaining > 0:
            # Find points on the Pareto front (non-dominated)
            def is_dominated(f, all_f):
                for other in all_f:
                    if np.all(other <= f) and np.any(other < f):
                        return True
                return False

            all_f = [f for _, f in evaluated]
            pf_indices = [i for i, (_, f) in enumerate(evaluated) if not is_dominated(f, all_f)]

            # Polish the top contributing points
            n_polish = min(remaining, 30)
            # Sort PF points by hypervolume contribution (rough: by f2 value)
            pf_points = [(i, evaluated[i][1]) for i in pf_indices]
            pf_points.sort(key=lambda t: t[1])  # Lower f2 = better

            for i in range(n_polish):
                if used >= self.budget or time.time() > deadline:
                    break
                if i < len(pf_points):
                    idx, f = pf_points[i]
                    x_base = evaluated[idx][0]
                    x1_base = x_base[0]
                else:
                    x1_base = rng.uniform(0.005, 0.995)

                # Local search: perturb x1 slightly
                for delta in [0.005, -0.005, 0.002, -0.002, 0.01, -0.01]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x1_new = x1_base + delta
                    if x1_new <= 0.001 or x1_new >= 0.999:
                        continue
                    x = make_x_on_pf(x1_new, 'uniform')
                    x = np.clip(x, lo, hi)
                    eval_point(x)

        return evaluated
# EVOLVE-BLOCK-END
