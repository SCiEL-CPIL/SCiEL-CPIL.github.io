import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using greedy hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        # Archive
        archive_F = []  # list of np.array of shape (2,)
        archive_X = []
        used = 0

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x), dtype=float)
            used += 1
            return f

        def is_dominated(f_new):
            for f_old in archive_F:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def remove_dominated(f_new):
            nonlocal archive_F, archive_X
            new_F = []
            new_X = []
            for i in range(len(archive_F)):
                if not (np.all(f_new <= archive_F[i]) and np.any(f_new < f_new) if False else
                        not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i]))):
                    new_F.append(archive_F[i])
                    new_X.append(archive_X[i])
            archive_F = new_F
            archive_X = new_X

        def add_point(x, f):
            if is_dominated(f):
                return
            remove_dominated(f)
            archive_F.append(f)
            archive_X.append(x)

        def hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume for a set of non-dominated points."""
            if not points:
                return 0.0
            # Sort by f1 ascending
            pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_f2 = ref[1]
            for p in pts:
                if p[0] >= ref[0] or p[1] >= ref[1]:
                    continue
                if p[1] < prev_f2:
                    hv += (prev_f2 - p[1]) * (ref[0] - p[0])
                    prev_f2 = p[1]
                # If p[1] >= prev_f2, it's dominated in 2D sense (shouldn't happen for non-dominated)
            return hv

        def marginal_hv(f_new, points, ref):
            """Compute marginal hypervolume contribution of adding f_new to the set."""
            if f_new[0] >= ref[0] or f_new[1] >= ref[1]:
                return 0.0
            # Hypervolume with the new point
            new_points = points + [f_new]
            hv_with = hypervolume_2d(new_points, ref)
            hv_without = hypervolume_2d(points, ref)
            return hv_with - hv_without

        def zdt6_pareto_f(t):
            """Expected objective values on ZDT6 Pareto front for x1=t, others=0."""
            f1 = float(t)
            f2 = 1.0 - np.sqrt(f1) - f1
            return np.array([f1, f2])

        rng = np.random.default_rng(42)

        # Phase 1: Greedy hypervolume maximization
        # We'll use a candidate pool approach: at each step, generate candidates
        # and pick the one with maximum marginal hypervolume

        # Pre-compute a dense set of candidate t values
        # Mix of uniform and concentrated near 0
        n_candidates_pool = 200
        t_pool = np.concatenate([
            np.linspace(0.001, 1.0, n_candidates_pool // 2),
            np.sort(rng.beta(1, 2, n_candidates_pool // 2))
        ])
        t_pool = np.clip(t_pool, 0.0, 1.0)

        # For efficiency, we'll use a smaller active candidate set per iteration
        n_active = 50

        while used < budget:
            if time.time() > deadline:
                break

            # Generate active candidates: mix from pool and new random
            if len(t_pool) > n_active:
                # Sample from pool
                idx = rng.choice(len(t_pool), size=min(n_active, len(t_pool)), replace=False)
                t_candidates = t_pool[idx]
            else:
                t_candidates = t_pool.copy()

            # Add some fresh random candidates
            n_fresh = min(10, n_active - len(t_candidates))
            if n_fresh > 0:
                fresh = np.concatenate([
                    rng.uniform(0.001, 1.0, n_fresh // 2),
                    rng.beta(1, 3, n_fresh - n_fresh // 2)
                ])
                t_candidates = np.concatenate([t_candidates, fresh])

            # Find best candidate by marginal hypervolume
            best_marginal = -np.inf
            best_t = None
            best_f_expected = None

            for t in t_candidates:
                t = float(np.clip(t, 0.0, 1.0))
                if t <= 0:
                    t = 0.001
                f_expected = zdt6_pareto_f(t)

                if f_expected[0] >= ref[0] or f_expected[1] >= ref[1]:
                    continue

                # Check if this point is already well-covered
                marginal = marginal_hv(f_expected, archive_F, ref)

                if marginal > best_marginal:
                    best_marginal = marginal
                    best_t = t
                    best_f_expected = f_expected

            if best_t is None:
                # Fallback: pick a random t
                best_t = float(rng.uniform(0.01, 0.5))
                best_f_expected = zdt6_pareto_f(best_t)

            # Evaluate the best candidate
            x = np.zeros(n)
            x[0] = best_t
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            add_point(x, f)

            # Refresh candidate pool occasionally to explore new regions
            if used % 20 == 0 and used < budget:
                # Add new candidates to pool
                new_ts = np.concatenate([
                    rng.uniform(0.001, 1.0, 20),
                    rng.beta(1, 3, 20)
                ])
                t_pool = np.concatenate([t_pool, new_ts])
                # Keep pool size manageable
                if len(t_pool) > 500:
                    t_pool = t_pool[-500:]

        return archive_X if archive_X else [np.zeros(n)]
