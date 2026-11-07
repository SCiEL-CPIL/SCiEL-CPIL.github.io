"""Multi-objective optimizer on WFG9 using MOEA/D with neighborhood mating and external archive."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using MOEA/D with:
    - Tchebycheff scalarization with adaptive ideal/nadir points
    - Neighborhood-based parent selection
    - External archive for non-dominated solutions
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        rng = np.random.RandomState(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Generate reference points on the simplex using systematic sampling
        # For 3 objectives, use a grid with n_grid divisions
        n_grid = 6
        ref_points = []
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                w1 = i / n_grid
                w2 = j / n_grid
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-10:
                    ref_points.append([w1, w2, max(0.0, w3)])

        ref_points = np.array(ref_points)
        ref_points = ref_points / ref_points.sum(axis=1, keepdims=True)
        n_sub = len(ref_points)

        # Build neighborhood structure: k nearest reference points by Euclidean distance
        k_neighbors = 5
        neighborhoods = []
        for i in range(n_sub):
            dists = np.linalg.norm(ref_points - ref_points[i], axis=1)
            dists[i] = np.inf  # exclude self
            neighbors = np.argsort(dists)[:k_neighbors]
            neighborhoods.append(list(neighbors))

        # Initialize population
        X = np.zeros((n_sub, self.dim))
        for i in range(n_sub):
            X[i] = lo + span * rng.random(self.dim)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_sub

        # Track best solution per subproblem
        best_X = X.copy()
        best_F = F.copy()

        # External archive: list of non-dominated solutions
        archive_X = [X[i].copy() for i in range(n_sub)]
        archive_F = [F[i].copy() for i in range(n_sub)]

        # Track ideal and nadir points
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)

        def update_archive(x_new, f_new):
            """Update external archive with new solution if non-dominated."""
            non_dominated = True
            for i in range(len(archive_F)):
                # If archive[i] dominates new, new is dominated
                if np.all(archive_F[i] <= f_new) and np.any(archive_F[i] < f_new):
                    non_dominated = False
                    break
                # If new dominates archive[i], remove archive[i]
                if np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i]):
                    archive_X.pop(i)
                    archive_F.pop(i)
                    i -= 1

            if non_dominated:
                archive_X.append(x_new.copy())
                archive_F.append(f_new.copy())

        # Initialize archive
        archive_X = []
        archive_F = []
        for i in range(n_sub):
            update_archive(X[i], F[i])

        # Tchebycheff scalarization with ideal and nadir
        def tchebycheff(f, w, z_ideal, z_nadir):
            """
            Tchebycheff scalarization:
            min max_i (w_i * |f_i - z_i^*| / (z_i^max - z_i^*))
            """
            range_f = z_nadir - z_ideal
            range_f = np.where(range_f > 1e-10, range_f, 1.0)
            w_safe = np.where(w > 1e-10, w, 1e-10)
            return np.max(w_safe * np.abs(f - z_ideal) / range_f)

        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            for i in range(n_sub):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Select parents from neighborhood
                neighbors = neighborhoods[i]
                if len(neighbors) >= 2:
                    p1_idx = neighbors[rng.randint(len(neighbors))]
                    remaining = [n for n in neighbors if n != p1_idx]
                    p2_idx = remaining[rng.randint(len(remaining))]
                else:
                    p1_idx = rng.randint(n_sub)
                    p2_idx = rng.randint(n_sub)

                # DE mutation with adaptive scale
                F_scale = 0.5 + 0.3 * rng.random()
                child = best_X[i] + F_scale * (best_X[p1_idx] - best_X[p2_idx])

                # Simulated binary crossover-like: blend with current best
                crossover_rate = 0.5
                for j in range(self.dim):
                    if rng.random() < crossover_rate:
                        child[j] = best_X[i][j]

                # Add small Gaussian noise for exploration
                noise_scale = 0.01 * span
                child += noise_scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update ideal and nadir points
                ideal = np.minimum(ideal, fc)
                nadir = np.maximum(nadir, fc)

                # Evaluate Tchebycheff fitness
                fitness_new = tchebycheff(fc, ref_points[i], ideal, nadir)
                fitness_old = tchebycheff(best_F[i], ref_points[i], ideal, nadir)

                # Check domination
                dominates_old = np.all(fc <= best_F[i]) and np.any(fc < best_F[i])

                # Replace if better in Tchebycheff sense or dominates
                if fitness_new < fitness_old or dominates_old:
                    best_X[i] = child
                    best_F[i] = fc

                # Update external archive
                update_archive(child, fc)

        return best_X
# EVOLVE-BLOCK-END
