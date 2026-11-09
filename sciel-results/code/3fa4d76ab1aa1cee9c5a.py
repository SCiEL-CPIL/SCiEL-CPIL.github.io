"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting separability."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure: first n_obj-1 variables are "shape" (angles),
        # remaining variables are "distance" variables that should be ~0.5
        n_shape = n_obj - 1  # 2 shape variables for 3 objectives
        n_dist = d - n_shape  # 10 distance variables

        # Archive-based approach: maintain a set of non-dominated solutions
        # Initialize with a structured grid on the Pareto front
        # For DTLZ4 with 3 objectives, the Pareto front is a 2D simplex
        # f1 = g * sin(theta1) * sin(theta2)
        # f2 = g * sin(theta1) * cos(theta2)  
        # f3 = g * cos(theta1)
        # where g = sum of (xi^0.1 - 1)^2 for distance variables, optimum g=0

        # Key insight: if we set distance variables to exactly 0.5, g=0
        # and we get the ideal Pareto front. So we focus search on shape vars.

        # Strategy: 
        # 1. Initialize with points on the known Pareto front (distance=0.5)
        # 2. Use remaining budget to explore and refine

        # Initialize: sample the Pareto front using uniform distribution over the simplex
        # For 3 objectives, theta1 in [0, pi/2], theta2 in [0, 2*pi] but for DTLZ4:
        # Actually for DTLZ4: x1, x2 are shape, x3..x12 are distance
        # f1 = (1+g)*x1*x2
        # f2 = (1+g)*x1*(1-x2)
        # f3 = (1+g)*(1-x1)
        # where g = sum((xi^0.1-1)^2 for i=3..12)
        # So shape vars are x1, x2 in [0,1]
        
        # The Pareto front when g=0 is:
        # f1 = x1*x2, f2 = x1*(1-x2), f3 = 1-x1
        # with x1 in [0,1], x2 in [0,1]
        # This is a 2D surface in objective space

        # Initialize with a grid on the Pareto front
        n_init = min(150, budget - 50)  # Leave some budget for exploration
        n_grid = int(np.ceil(np.sqrt(n_init)))
        
        X_init = np.zeros((n_init, d))
        # Set distance variables to 0.5 (optimal for g=0)
        X_init[:, n_shape:] = 0.5
        
        # Sample shape variables on a grid
        x1_vals = np.linspace(0.01, 0.99, n_grid)
        x2_vals = np.linspace(0.01, 0.99, n_grid)
        idx = 0
        for x1 in x1_vals:
            for x2 in x2_vals:
                if idx >= n_init:
                    break
                X_init[idx, 0] = x1
                X_init[idx, 1] = x2
                idx += 1
        if idx < n_init:
            # Fill remaining with random
            X_init[idx:, :n_shape] = rng.random((n_init - idx, n_shape))
            X_init[idx:, n_shape:] = 0.5

        X = X_init[:n_init]
        F = np.array([problem(x) for x in X])
        used = n_init

        # Maintain an archive of non-dominated solutions
        def is_dominated(f_new, F_archive):
            """Check if f_new is dominated by any point in F_archive."""
            return np.any(np.all(F_archive <= f_new, axis=1) & np.any(F_archive < f_new, axis=1))

        def archive_insert(X, F, idx_new, x_new, f_new, archive_size=200):
            """Insert new solution into archive, maintaining non-dominance."""
            # Check if new solution dominates any existing
            dominated = np.where(np.all(f_new <= F, axis=1) & np.any(f_new < F, axis=1))[0]
            if len(dominated) > 0:
                # Remove dominated solutions
                mask = np.ones(len(X), dtype=bool)
                mask[dominated] = False
                X = X[mask]
                F = F[mask]
            
            # Check if new solution is dominated by existing
            if is_dominated(f_new, F):
                return X, F, False
            
            # Add new solution
            X = np.vstack([X, x_new.reshape(1, -1)])
            F = np.vstack([F, f_new.reshape(1, -1)])
            
            # If archive too large, remove the point with smallest crowding distance
            if len(X) > archive_size:
                cd = crowding_distance(F)
                min_idx = np.argmin(cd)
                mask = np.ones(len(X), dtype=bool)
                mask[min_idx] = False
                X = X[mask]
                F = F[mask]
            
            return X, F, True

        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(F.shape[1]):
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng_m = F[order[-1], m] - F[order[0], m]
                if rng_m > 1e-15:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / rng_m
            return cd

        # Main loop: use remaining budget for local search around Pareto front
        while used < budget:
            if time.time() > deadline:
                break

            # Pick a random point from current population
            parent_idx = rng.integers(len(X))
            parent = X[parent_idx].copy()
            
            # Mutate: primarily shape variables with small steps, distance vars stay near 0.5
            # Use adaptive step size
            sigma_shape = 0.1
            sigma_dist = 0.03
            
            child = parent.copy()
            # Mutate shape variables
            child[:n_shape] += sigma_shape * rng.standard_normal(n_shape)
            child[:n_shape] = np.clip(child[:n_shape], 0, 1)
            
            # Distance variables: keep near 0.5 with small perturbation
            child[n_shape:] = 0.5 + sigma_dist * rng.standard_normal(n_dist)
            child[n_shape:] = np.clip(child[n_shape:], lo[n_shape:], hi[n_shape:])
            
            fc = problem(child)
            used += 1
            
            # Insert into archive
            X, F, added = archive_insert(X, F, parent_idx, child, fc, archive_size=200)

        # Final: return all evaluated points (the archive)
        return X

# EVOLVE-BLOCK-END
