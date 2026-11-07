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
        n_sub = 25  # 25 subproblems

        # Generate reference points on the simplex (uniformly distributed)
        # Using a simple lattice: for 3 objectives, points (w1, w2, w3) with wi >= 0, sum = 1
        # We'll use a stratified approach
        ref_points = []
        # For 3 objectives, we can use a grid
        # Number of divisions along each axis
        # We want approximately n_sub points
        # Use a simple approach: generate points and normalize
        # Better: use a fixed set of well-distributed points

        # For 3 objectives, use a set of points that cover the simplex
        # We'll use a combination of vertices, edge midpoints, and interior points
        # Or use a simple grid and normalize

        # Simple approach: use a grid on the 2-simplex
        n_grid = 5  # 5x5 grid gives 21 points for 3 objectives (triangular)
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                w1 = i / n_grid
                w2 = j / n_grid
                w3 = 1 - w1 - w2
                if w3 >= -1e-10:
                    ref_points.append([w1, w2, max(0, w3)])

        ref_points = np.array(ref_points)
        # Normalize to sum to 1
        ref_points = ref_points / ref_points.sum(axis=1, keepdims=True)
        n_sub = len(ref_points)

        # Initialize population: one solution per subproblem
        X = np.zeros((n_sub, self.dim))
        for i in range(n_sub):
            # Initialize near the center with some spread
            X[i] = (lo + hi) / 2 + 0.1 * span * rng.standard_normal(self.dim)
            X[i] = np.clip(X[i], lo, hi)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_sub

        # Track best solution per subproblem
        best_X = X.copy()
        best_F = F.copy()

        # Tchebycheff scalarization
        def tchebycheff(f, w, rho=0.05):
            """
            Tchebycheff scalarization:
            min max_i (w_i * |f_i - z_i^*|)
            where z^* is the ideal point
            """
            # We'll use a simplified version without ideal point tracking
            # Use: max_i (w_i * f_i)
            return np.max(w * f)

        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # For each subproblem, generate a child using DE-like mutation
            for i in range(n_sub):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Select parents from neighboring subproblems
                # Find 2 neighbors (or random if not enough)
                neighbors = list(range(n_sub))
                neighbors.remove(i)
                p1_idx = rng.choice(neighbors)
                neighbors.remove(p1_idx)
                p2_idx = rng.choice(neighbors)

                # DE mutation
                F_scale = 0.5 + 0.3 * rng.random()
                child = best_X[i] + F_scale * (best_X[p1_idx] - best_X[p2_idx])

                # Crossover with current best
                crossover_rate = 0.5
                for j in range(self.dim):
                    if rng.random() < crossover_rate:
                        child[j] = best_X[i][j]

                # Add small noise
                noise_scale = 0.02 * span
                child += noise_scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Evaluate Tchebycheff fitness for this subproblem
                fitness_new = tchebycheff(fc, ref_points[i])
                fitness_old = tchebycheff(best_F[i], ref_points[i])

                # Replace if better in Tchebycheff sense, or if it dominates
                # Check domination
                dominates_old = np.all(fc <= best_F[i]) and np.any(fc < best_F[i])
                dominated_by_old = np.all(best_F[i] <= fc) and np.any(best_F[i] < fc)

                if fitness_new < fitness_old or dominates_old:
                    best_X[i] = child
                    best_F[i] = fc

        return best_X
# EVOLVE-BLOCK-END
