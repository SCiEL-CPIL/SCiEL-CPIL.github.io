"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast evolutionary multi-objective optimizer for WFG7 with weighted-sum guidance and polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 35

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Precompute some weight vectors for guided search
        # Use a set of diverse weight vectors to encourage spread
        n_weights = 12
        # Generate diverse weight vectors on the simplex
        weight_vectors = []
        # Add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        # Add uniform
        weight_vectors.append(np.ones(self.n_obj) / self.n_obj)
        # Add random points on simplex
        while len(weight_vectors) < n_weights:
            w = rng.random(self.n_obj)
            w /= w.sum()
            weight_vectors.append(w)
        weight_vectors = np.array(weight_vectors)

        # Track best points found (all non-dominated so far)
        all_points_X = [X.copy()]
        all_points_F = [F.copy()]

        def get_non_dominated(F_mat):
            """Return indices of non-dominated points and dominance count per point."""
            N = F_mat.shape[0]
            if N == 0:
                return np.array([], dtype=int), np.zeros(0, dtype=int)
            F_j = F_mat[:, np.newaxis, :]  # (N, 1, M)
            F_i = F_mat[np.newaxis, :, :]  # (1, N, M)
            le = F_j <= F_i
            lt = F_j < F_i
            all_le = np.all(le, axis=2)
            any_lt = np.any(lt, axis=2)
            dom = all_le & any_lt
            np.fill_diagonal(dom, False)
            dominated = np.any(dom, axis=0)
            dom_count = np.sum(dom, axis=0)  # How many points dominate each point
            nd_indices = np.where(~dominated)[0]
            return nd_indices, dom_count

        # Main loop
        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break

            progress = used / self.budget

            # Find non-dominated set
            nd_indices, dom_count = get_non_dominated(F)

            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # Choose operation: crossover, guided mutation, or pure mutation
            r = rng.random()
            if len(nd_indices) >= 2 and r < 0.5:
                # Crossover between two non-dominated parents
                p1_idx = nd_indices[rng.integers(len(nd_indices))]
                p2_idx = nd_indices[rng.integers(len(nd_indices))]
                parent1 = X[p1_idx]
                parent2 = X[p2_idx]

                # SBX-like crossover with uniform blend
                alpha = rng.random(self.dim)
                child = alpha * parent1 + (1 - alpha) * parent2
            elif r < 0.85:
                # Weighted-sum guided mutation
                parent_idx = nd_indices[rng.integers(len(nd_indices))]
                parent = X[parent_idx]

                # Pick a random weight vector
                w = weight_vectors[rng.integers(n_weights)]

                # Compute gradient-like direction: perturb to reduce weighted sum
                # Use a small step in direction that decreases weighted sum
                # Approximate: try a few directions and pick best
                n_trials = 3
                best_child = parent.copy()
                best_wsum = np.dot(F[parent_idx], w)

                step_size = 0.12 * (1 - 0.6 * progress)
                for _ in range(n_trials):
                    direction = rng.standard_normal(self.dim)
                    candidate = parent + step_size * span * direction
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1
                    wsum = np.dot(fc, w)
                    if wsum < best_wsum:
                        best_wsum = wsum
                        best_child = candidate
                        best_fc = fc

                child = best_child
                fc = best_fc
            else:
                # Pure random mutation
                parent_idx = nd_indices[rng.integers(len(nd_indices))]
                parent = X[parent_idx]
                step_size = 0.10 * (1 - 0.5 * progress)
                child = parent + step_size * span * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1

            # If we used trials, fc is already set; otherwise evaluate
            if 'fc' not in dir() or r < 0.5:
                fc = problem(child)
                used += 1

            # Replacement strategy
            # Find the point to replace: prefer dominated points, then worst weighted sum
            # Compute dominance counts again (may have changed if we're replacing)
            # Actually, we should decide replacement before evaluating child to save time,
            # but we need child's F to check if it's dominated. Let's do a simpler approach:
            # Replace the point with highest dom_count, breaking ties by sum of objectives.

            # Recompute for current F (before adding child)
            _, dom_count_cur = get_non_dominated(F)
            F_sums = np.sum(F, axis=1)

            # Score for replacement: higher is worse
            # Use dom_count as primary, F_sum as secondary
            replace_score = dom_count_cur.astype(float) * 1000 + F_sums
            replace_idx = int(np.argmax(replace_score))

            # Check if child is dominated by current population
            le_c = F <= fc[np.newaxis, :]
            lt_c = F < fc[np.newaxis, :]
            doms_child = np.all(le_c, axis=1) & np.any(lt_c, axis=1)
            child_is_dominated = np.any(doms_child)

            # If child is not dominated, always accept (replace worst)
            # If child is dominated, only accept if it's better than the point we're replacing
            if not child_is_dominated:
                X[replace_idx] = child
                F[replace_idx] = fc
            else:
                # Compare child with the point being replaced
                child_sum = np.sum(fc)
                replaced_sum = F_sums[replace_idx]
                if child_sum < replaced_sum:
                    X[replace_idx] = child
                    F[replace_idx] = fc
                # Otherwise, don't replace (keep population stable)

        # Final polish: use remaining budget to refine best points
        if used < self.budget:
            remaining = self.budget - used
            # Get current non-dominated set
            nd_indices, _ = get_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # Sort by sum, take best few
            nd_sums = np.sum(F[nd_indices], axis=1)
            order = np.argsort(nd_sums)
            top_k = min(len(nd_indices), 8)
            polish_indices = nd_indices[order[:top_k]]

            for _ in range(remaining):
                if time.time() > deadline:
                    break

                # Pick a random point from polish set
                pi = polish_indices[rng.integers(len(polish_indices))]
                parent = X[pi]
                fc_parent = F[pi]

                # Small local step
                step_size = 0.03
                child = parent + step_size * span * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1

                # Accept if better in sum or non-dominated
                child_sum = np.sum(fc)
                parent_sum = np.sum(fc_parent)

                # Check if child is dominated by parent
                le_p = fc_parent <= fc
                lt_p = fc_parent < fc
                parent_doms_child = np.all(le_p) and np.any(lt_p)

                if not parent_doms_child and child_sum <= parent_sum:
                    X[pi] = child
                    F[pi] = fc

        return X
# EVOLVE-BLOCK-END
