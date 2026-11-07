"""Rewritten multi-objective optimizer for WFG7 with deadline guard."""
import time
import numpy as np


class Optimizer:
    """Evolutionary multi-objective optimizer with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        rng = np.random.default_rng(42)

        # Population size - larger for better coverage
        pop_size = min(60, max(20, self.budget // 4))

        # Initialize population with diverse spread
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best non-dominated set as list of (x, f) tuples
        best_set = []

        def is_dominated(f_new, F_existing):
            """Check if f_new is dominated by any point in F_existing."""
            if len(F_existing) == 0:
                return False
            F_arr = np.array(F_existing)
            dominates = np.all(F_arr <= f_new, axis=1) & np.any(F_arr < f_new, axis=1)
            return np.any(dominates)

        def dominates_any(f_new, F_existing):
            """Check if f_new dominates any point in F_existing."""
            if len(F_existing) == 0:
                return False
            F_arr = np.array(F_existing)
            dominated = np.all(f_new <= F_arr, axis=1) & np.any(f_new < F_arr, axis=1)
            return np.any(dominated)

        def update_best_set(x, f):
            """Add point to best_set if not dominated, remove dominated points."""
            # Check if new point is dominated by any existing
            if is_dominated(f, [ef for _, ef in best_set]):
                return
            # Remove any existing points dominated by new point
            new_best = []
            for ex, ef in best_set:
                if not (np.all(f <= ef) and np.any(f < ef)):
                    new_best.append((ex, ef))
            new_best.append((x, f))
            best_set[:] = new_best

        # Initial update - use efficient batch approach
        for i in range(len(X)):
            update_best_set(X[i], F[i])

        def compute_hv_contribution(f, other_points, ref):
            """Compute approximate HV contribution of a single point."""
            # Simple approximation: volume of box from f to ref,
            # minus overlap with other points
            if f[0] >= ref[0] or f[1] >= ref[1] or f[2] >= ref[2]:
                return 0.0
            
            box_vol = (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
            
            # Subtract overlap with other non-dominated points
            overlap = 0.0
            for of in other_points:
                if of[0] >= ref[0] or of[1] >= ref[1] or of[2] >= ref[2]:
                    continue
                # Intersection of boxes
                ix = max(0, min(ref[0], of[0]) - max(f[0], of[0]))
                iy = max(0, min(ref[1], of[1]) - max(f[1], of[1]))
                iz = max(0, min(ref[2], of[2]) - max(f[2], of[2]))
                overlap += ix * iy * iz
            
            return max(0, box_vol - overlap)

        def get_nondominated_indices(F_mat):
            """Vectorized non-dominated set computation."""
            n = len(F_mat)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                # Check if any other point dominates i
                others = np.delete(F_mat, i, axis=0)
                if len(others) == 0:
                    continue
                dominates_i = np.all(others <= F_mat[i], axis=1) & np.any(others < F_mat[i], axis=1)
                if np.any(dominates_i):
                    nondom[i] = False
            return np.where(nondom)[0]

        # Main evolution loop
        ref = np.array([3.0, 5.0, 7.0])
        
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated points from current population (vectorized)
            nondom_idx = get_nondominated_indices(F)
            if len(nondom_idx) == 0:
                nondom_idx = np.arange(len(F))

            # Select parents: prefer points with high HV contribution
            # Use tournament selection among non-dominated
            if len(nondom_idx) >= 2:
                # Pick 2 random candidates, select the one with better HV contribution
                candidates = rng.choice(nondom_idx, size=min(4, len(nondom_idx)), replace=False)
                hv_scores = []
                for c in candidates:
                    other_f = [F[j] for j in nondom_idx if j != c]
                    hv_scores.append(compute_hv_contribution(F[c], other_f, ref))
                best_cand = candidates[np.argmax(hv_scores)]
                second_cand = candidates[np.argmin(hv_scores)]
                p1_idx = best_cand
                p2_idx = second_cand
            else:
                p1_idx = nondom_idx[0]
                p2_idx = nondom_idx[-1] if len(nondom_idx) > 1 else nondom_idx[0]

            p1 = X[p1_idx]
            p2 = X[p2_idx]

            # Crossover: BLX-Alpha style for better exploration
            alpha = 0.5
            child = np.zeros(self.dim)
            for d in range(self.dim):
                if p1[d] < p2[d]:
                    child[d] = p1[d] - alpha * (p2[d] - p1[d]) * rng.random()
                    child[d] = max(child[d], lo[d])
                else:
                    child[d] = p2[d] - alpha * (p1[d] - p2[d]) * rng.random()
                    child[d] = max(child[d], lo[d])
                child[d] = min(child[d], hi[d])

            # Adaptive mutation: larger early, smaller later
            progress = used / self.budget
            sigma = 0.25 * (1.0 - 0.7 * progress)
            # Use Cauchy-like heavy-tailed mutation for occasional large jumps
            noise = rng.standard_normal(self.dim)
            if rng.random() < 0.1:
                noise *= 2.0  # occasional larger jump
            child = child + sigma * span * noise
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the point with lowest HV contribution in population
            # Among non-dominated points, find the one with least HV contribution
            if len(nondom_idx) > 0:
                hv_contribs = []
                for ni in nondom_idx:
                    other_f = [F[j] for j in nondom_idx if j != ni]
                    hv_contribs.append(compute_hv_contribution(F[ni], other_f, ref))
                worst_ni = nondom_idx[np.argmin(hv_contribs)]
                replace_idx = worst_ni
            else:
                # Replace worst by sum
                replace_idx = int(np.argmax(np.sum(F, axis=1)))

            X[replace_idx] = child
            F[replace_idx] = fc

            # Update best set
            update_best_set(child, fc)

        # Final local search polish on non-dominated points
        if time.time() < deadline - 15:
            nondom_idx = get_nondominated_indices(F)
            # Sort by HV contribution, polish the most valuable ones first
            if len(nondom_idx) > 0:
                hv_contribs = []
                for ni in nondom_idx:
                    other_f = [F[j] for j in nondom_idx if j != ni]
                    hv_contribs.append(compute_hv_contribution(F[ni], other_f, ref))
                polished = nondom_idx[np.argsort(hv_contribs)[::-1]]
            else:
                polished = np.arange(len(F))

            for idx in polished:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Adaptive local search with decreasing step size
                base_step = 0.03
                for attempt in range(5):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    step = base_step * (1.0 - 0.5 * attempt / 5.0)
                    pert = X[idx] + step * span * rng.standard_normal(self.dim)
                    pert = np.clip(pert, lo, hi)
                    fp = problem(pert)
                    used += 1

                    # Check if perturbation improves on original
                    if not (np.all(F[idx] <= fp) and np.any(F[idx] < fp)):
                        # Not dominated by original - add to best set
                        update_best_set(pert, fp)
                        # Update population if better in sum
                        if np.sum(fp) < np.sum(F[idx]):
                            X[idx] = pert
                            F[idx] = fp
                            base_step *= 0.7  # shrink step if improving

        return X
