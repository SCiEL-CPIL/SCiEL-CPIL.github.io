"""Multi-objective optimizer for RE21 with Halton init, hypervolume-guided search, and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 295.0  # safety margin before 330s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Halton sequence initialization for better coverage
        def halton(n, dim):
            primes = []
            candidate = 2
            while len(primes) < dim:
                is_prime = True
                for p in primes:
                    if p * p > candidate:
                        break
                    if candidate % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(candidate)
                candidate += 1

            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for d in range(dim):
                    f = 1.0
                    result = 0.0
                    idx = i
                    while idx > 0:
                        f /= primes[d]
                        result += f * (idx % primes[d])
                        idx //= primes[d]
                    points[i - 1, d] = result
            return points

        # Phase 1: Halton initialization
        n_init = min(60, self.budget // 3)
        halton_pts = halton(n_init, self.dim)
        X_init = lo + (hi - lo) * halton_pts
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        X = X_init.copy()
        F = F_init.copy()

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return np.array(nd, dtype=int)

        def hypervolume_2d(points, ref):
            """Compute hypervolume for 2D points relative to reference point."""
            # Filter points that are better than reference
            valid = np.all(points < ref, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for p in pts:
                if p[1] < prev_y:
                    hv += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
                else:
                    break
            return hv

        def hypervolume_contribution(F, idx, ref):
            """Compute hypervolume contribution of point idx."""
            # Hypervolume with point
            hv_with = hypervolume_2d(F, ref)
            # Hypervolume without point
            mask = np.ones(len(F), dtype=bool)
            mask[idx] = False
            hv_without = hypervolume_2d(F[mask], ref)
            return hv_with - hv_without

        ref_point = np.array([3000.0, 0.06])

        # Phase 2: Main search loop with hypervolume-guided selection
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated set
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Compute hypervolume contributions for non-dominated points
            nd_F = F[nd_idx]
            n_nd = len(nd_idx)

            if n_nd <= 100:
                # Compute contributions for all non-dominated points
                contributions = np.zeros(n_nd)
                for i in range(n_nd):
                    global_idx = nd_idx[i]
                    contributions[i] = hypervolume_contribution(F, global_idx, ref_point)
            else:
                # Approximate: use distance to reference as proxy
                contributions = np.sum(np.maximum(0, ref_point - nd_F), axis=1)

            # Select parent based on hypervolume contribution (inverse: lower contribution = more important)
            # Actually, we want to explore areas where adding a point would increase HV the most
            # So we should select parents that are in "gaps" - use a different strategy
            # Select parent with random weighting toward less-dominating points
            weights = 1.0 / (contributions + 1e-10)
            weights /= weights.sum()
            p_local = rng.choice(n_nd, p=weights)
            parent = F[nd_idx[p_local]]
            parent_x = X[nd_idx[p_local]]

            # Adaptive mutation
            step_size = 0.15 * (hi - lo) * rng.standard_normal(self.dim)
            child_x = parent_x + step_size
            child_x = np.clip(child_x, lo, hi)

            fc = problem(child_x)
            used += 1

            # Update population: replace point with lowest hypervolume contribution
            if n_nd <= 100:
                worst_local = np.argmin(contributions)
            else:
                worst_local = np.argmax(np.sum(nd_F, axis=1))
            worst_global = nd_idx[worst_local]

            # Only replace if child is non-dominated or improves HV
            if not any(dominates(F[j], fc) for j in range(len(F)) if j != worst_global):
                X[worst_global] = child_x
                F[worst_global] = fc
            else:
                # Keep old point, but still record evaluation
                pass

            # Occasionally do a larger random jump for exploration
            if rng.random() < 0.08 and used < self.budget:
                if time.time() > deadline:
                    break
                big_child = lo + (hi - lo) * rng.random(self.dim)
                big_fc = problem(big_child)
                used += 1
                # Replace worst overall point
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = big_child
                F[worst_idx] = big_fc

        # Phase 3: Local polish on non-dominated points
        nd_idx = non_dominated_indices(F)
        if len(nd_idx) > 0 and time.time() < deadline - 10:
            # Polish each non-dominated point with a local optimizer
            # Use a weighted sum approach to find nearby improvements
            for idx in nd_idx[:20]:  # Limit to 20 to save time
                if time.time() > deadline - 5:
                    break
                x0 = X[idx].copy()
                f0 = F[idx].copy()

                # Try a few different weight vectors
                for w in [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5]),
                          np.array([0.8, 0.2]), np.array([0.2, 0.8])]:
                    if time.time() > deadline - 5:
                        break

                    def objective(x):
                        fx = problem(x)
                        return float(np.dot(w, fx))

                    try:
                        res = minimize(objective, x0, method='Nelder-Mead',
                                       options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                        if res.success and res.fun < np.dot(w, f0) - 1e-8:
                            # Check if it's truly better in the multi-objective sense
                            f_new = problem(res.x)
                            if dominates(f_new, f0):
                                X[idx] = res.x
                                F[idx] = f_new
                                x0 = res.x
                                f0 = f_new
                    except Exception:
                        pass

        return X
# EVOLVE-BLOCK-END
