# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using decomposition (Tchebycheff)
    with per-subproblem differential evolution and hypervolume-aware selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.RandomState(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Number of subproblems (reference points on the simplex)
        # For 3 objectives, use a reasonable number for good coverage
        n_sub = 25

        # Generate reference points on the 2-simplex (3 objectives)
        # Using a grid approach for uniform coverage
        ref_points = []
        # For 3 objectives, points (w1, w2, w3) with w1+w2+w3=1, wi>=0
        # Use a triangular grid
        for i in range(6):
            for j in range(6 - i):
                k = 5 - i - j
                if i + j + k == 5:
                    ref_points.append([i / 5.0, j / 5.0, k / 5.0])
        # Add center and edge midpoints for better coverage
        ref_points.append([1.0/3, 1.0/3, 1.0/3])
        ref_points.append([0.5, 0.5, 0.0])
        ref_points.append([0.5, 0.0, 0.5])
        ref_points.append([0.0, 0.5, 0.5])
        ref_points.append([0.0, 0.0, 1.0])
        ref_points.append([0.0, 1.0, 0.0])
        ref_points.append([1.0, 0.0, 0.0])
        ref_points = np.array(ref_points)
        n_sub = len(ref_points)

        # Initialize: one solution per subproblem using LHS-like spread
        X = np.zeros((n_sub, self.dim))
        F = np.zeros((n_sub, self.n_obj))

        # Latin Hypercube Sampling for initial population
        for j in range(self.dim):
            perm = rng.permutation(n_sub)
            X[np.arange(n_sub), j] = lo[j] + (hi[j] - lo[j]) * (perm + rng.uniform(size=n_sub)) / n_sub
        X = np.clip(X, lo, hi)

        used = 0
        for i in range(n_sub):
            F[i] = problem(X[i])
            used += 1

        def dominates(a, b):
            """True if a strictly dominates b."""
            return np.all(a <= b) and np.any(a < b)

        def tchebycheff_distance(F_i, weights):
            """Compute Tchebycheff distance for solution i with given weights."""
            max_f = np.max(F, axis=0)
            return np.max(weights * np.abs(F_i - max_f))

        # Assign initial solutions to subproblems by Tchebycheff distance
        # Each subproblem gets the solution with best Tchebycheff score for its weights

        # Track best solution per subproblem
        best_X = X.copy()
        best_F = F.copy()
        best_tch = np.zeros(n_sub)
        for i in range(n_sub):
            best_tch[i] = tchebycheff_distance(F[i], ref_points[i])

        # Main loop: DE per subproblem
        scale = 0.5
        crossover_rate = 0.5
        neighbor_range = 2  # Look at neighbors within this distance

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Process each subproblem in random order
            order = rng.permutation(n_sub)

            for sub in order:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Find neighbors: subproblems with similar weights
                # Use cosine similarity or distance in weight space
                dists = np.sum((ref_points - ref_points[sub]) ** 2, axis=1)
                # Sort by distance, take nearest neighbors (excluding self)
                sorted_idx = np.argsort(dists)
                neighbors = []
                for idx in sorted_idx:
                    if idx != sub:
                        neighbors.append(idx)
                    if len(neighbors) >= 3:
                        break

                if len(neighbors) < 2:
                    # Fallback: use random
                    neighbors = list(set(list(np.random.choice(n_sub, size=3, replace=False))) - {sub})

                # Select parents from neighbors
                # Pick 2 distinct neighbors
                if len(neighbors) >= 2:
                    p1_idx, p2_idx = rng.choice(neighbors, size=2, replace=False)
                else:
                    p1_idx = p2_idx = neighbors[0]

                # DE mutation: v = x[sub] + F * (x[p1] - x[p2])
                v = best_X[sub] + scale * (best_X[p1_idx] - best_X[p2_idx])

                # Crossover
                trial = v.copy()
                for j in range(self.dim):
                    if rng.random() < crossover_rate:
                        trial[j] = best_X[sub][j]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Check if trial is better for this subproblem
                tch_trial = tchebycheff_distance(f_trial, ref_points[sub])

                # Also check global dominance
                is_dominated = False
                for i in range(n_sub):
                    if i != sub and dominates(best_F[i], f_trial):
                        is_dominated = True
                        break

                # Update if better Tchebycheff distance and not dominated
                if tch_trial < best_tch[sub] and not is_dominated:
                    best_X[sub] = trial
                    best_F[sub] = f_trial
                    best_tch[sub] = tch_trial
                elif tch_trial < best_tch[sub]:
                    # Even if dominated, keep if significantly better locally
                    if tch_trial < best_tch[sub] * 0.9:
                        best_X[sub] = trial
                        best_F[sub] = f_trial
                        best_tch[sub] = tch_trial

        # Return all best solutions
        return best_X
# EVOLVE-BLOCK-END
