"""Multi-objective optimizer on RE21 using HypE-style marginal hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective search with stratified initialization and adaptive sigma."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([3000.0, 0.06])

        # Population size
        pop_size = min(30, max(10, budget // 6))

        # Stratified initialization for better spread
        # Use Sobol-like stratification
        def stratified_sample(n, d, lo, hi, rng):
            """Generate stratified random samples in [lo, hi]^d."""
            samples = np.zeros((n, d))
            for i in range(n):
                for j in range(d):
                    # Stratify each dimension
                    stratum = (i * d + j) % n
                    frac = (stratum + rng.random()) / n
                    samples[i, j] = lo[j] + (hi[j] - lo[j]) * frac
            return samples

        X = stratified_sample(pop_size, dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [x.copy() for x in X]
        all_F = [f.copy() for f in F]

        # Adaptive sigma
        sigma = 0.2  # initial step size relative to bounds

        def hypervolume_2d(points, ref):
            """Calculate 2D hypervolume for a set of points with given reference point.
            Only points strictly better than ref in every objective count."""
            if len(points) == 0:
                return 0.0
            # Filter points that are strictly better than ref in every objective
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x, y = pts[i]
                if y < prev_y:
                    hv += (ref[0] - x) * (prev_y - y)
                    prev_y = y
            return hv

        def marginal_hypervolume_contributions(F, ref):
            """Calculate marginal hypervolume contribution of each point.
            HV with point - HV without point."""
            n = len(F)
            if n == 0:
                return np.array([])
            contributions = np.zeros(n)
            for i in range(n):
                # HV with all points
                hv_all = hypervolume_2d(F, ref)
                # HV without point i
                F_without = np.delete(F, i, axis=0)
                hv_without = hypervolume_2d(F_without, ref)
                contributions[i] = hv_all - hv_without
            return contributions

        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Calculate marginal hypervolume contributions for selection
            # Use a subset if population is large for speed
            if pop_size <= 20:
                contribs = marginal_hypervolume_contributions(F, ref)
            else:
                # Approximate: use non-dominated points' contributions
                nd_idx = non_dominated_sort(F)
                contribs = np.zeros(pop_size)
                if len(nd_idx) > 0:
                    nd_F = F[nd_idx]
                    nd_contribs = marginal_hypervolume_contributions(nd_F, ref)
                    contribs[nd_idx] = nd_contribs
                # Add small random component for exploration
                contribs += 0.1 * np.random.RandomState(0).random(pop_size)

            # Select parent with probability proportional to contribution
            contribs = np.maximum(contribs, 1e-10)
            probs = contribs / contribs.sum()
            parent_idx = rng.choice(pop_size, p=probs)
            parent = X[parent_idx].copy()

            # Adaptive sigma: decrease over time
            progress = used / budget
            sigma_current = sigma * (1.0 - 0.5 * progress)

            # Mutate
            child = parent + sigma_current * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Track in all evaluations
            all_X.append(child.copy())
            all_F.append(fc.copy())

            # Update population: replace worst point by sum of objectives
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        # Final polish: do a few local refinements around best hypervolume contributors
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 10:
            # Find top contributors
            if len(F) <= 20:
                contribs = marginal_hypervolume_contributions(F, ref)
            else:
                nd_idx = non_dominated_sort(F)
                contribs = np.zeros(len(F))
                if len(nd_idx) > 0:
                    nd_F = F[nd_idx]
                    nd_contribs = marginal_hypervolume_contributions(nd_F, ref)
                    contribs[nd_idx] = nd_contribs
                contribs += 0.1 * np.random.RandomState(0).random(len(F))

            top_k = min(3, len(F))
            top_idx = np.argsort(contribs)[-top_k:]

            for _ in range(remaining):
                if time.time() > deadline - 2:
                    break
                idx = top_idx[rng.integers(len(top_idx))]
                parent = X[idx].copy()
                # Tighter step for polish
                child = parent + 0.05 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                all_X.append(child.copy())
                all_F.append(fc.copy())
                # Replace worst
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc

        return X
# EVOLVE-BLOCK-END
