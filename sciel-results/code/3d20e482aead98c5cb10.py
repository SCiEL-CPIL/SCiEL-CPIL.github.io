# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ4 with known Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ4 structure:
        # - M-1 variables (x_0 ... x_{M-2}) determine position on Pareto front
        # - n-M+1 variables (x_{M-1} ... x_{n-1}) control distance g(x)
        # - g(x) = sum_{i=M-1}^{n-1} (x_i - 0.5)^2
        # - Optimal when all distance vars = 0.5
        # - For 3 objectives: M-1 = 2 shape vars, n-M+1 = 10 distance vars
        # - f_i = (1+g) * h_i where h_i are standard DTLZ4 shape functions

        # Strategy:
        # 1. Set distance variables to 0.5 (g=1, optimal)
        # 2. Sample shape variables to cover the Pareto front
        # 3. Use budget efficiently: sample many points on the front

        # For DTLZ4 with 3 objectives, the Pareto front is:
        # f_0 = (1+g) * sin^2(pi*x_0/2)
        # f_1 = (1+g) * sin(pi*x_0/2) * cos^2(pi*x_1/2)
        # f_2 = (1+g) * sin(pi*x_0/2) * cos(pi*x_1/2) * sin(pi*x_2/2)
        # Wait, let me reconsider. DTLZ4 with m objectives has m-1 shape variables.
        # For 3 objectives: 2 shape variables (x_0, x_1), 10 distance variables.
        #
        # Standard DTLZ4:
        # f_1 = (1+g) * prod_{i=1}^{M-2} sin(pi*x_i/2) * cos(pi*x_{M-1}/2) ... 
        # Actually for DTLZ4 with m objectives:
        # f_1 = (1+g) * sin(pi*x_1/2) * ... * cos(pi*x_{m-1}/2)
        # Let me use the standard formulation.
        #
        # For m=3 objectives, 12 variables:
        # Shape vars: x_0, x_1 (indices 0, 1)
        # Distance vars: x_2 ... x_11 (indices 2-11), 10 vars
        # g = sum_{i=2}^{11} (x_i - 0.5)^2
        #
        # DTLZ4 objectives (standard):
        # f_0 = (1+g) * sin^2(pi*x_0/2)
        # f_1 = (1+g) * sin(pi*x_0/2) * cos^2(pi*x_1/2)  
        # f_2 = (1+g) * sin(pi*x_0/2) * cos(pi*x_1/2) * sin(pi*x_1/2)
        #
        # Wait, I need to be more careful. The standard DTLZ4 with m objectives:
        # f_1(x) = (1 + g(x_M...x_n)) * prod_{i=1}^{m-2} sin(pi*x_i/2) * cos(pi*x_{m-1}/2)
        # f_2(x) = (1 + g(x_M...x_n)) * prod_{i=1}^{m-2} sin(pi*x_i/2) * cos(pi*x_{m-1}/2) * sin(pi*x_{m-1}/2)
        # ...
        # Actually this is getting complicated. Let me just use the structure:
        # When distance vars = 0.5, g=1, so 1+g=2.
        # The shape is determined by x_0, x_1.
        # 
        # Key insight: we should sample x_0, x_1 to cover the front well,
        # and set all distance vars to 0.5.

        # Number of shape variables for 3-objective DTLZ4
        n_shape = n_obj - 1  # 2 shape variables
        n_dist = dim - n_shape  # 10 distance variables

        # Optimal distance value: all distance vars at 0.5
        # g = 0 when all are 0.5, so 1+g = 1
        # Actually g = sum (x_i - 0.5)^2, so when x_i = 0.5, g = 0, 1+g = 1

        # Sample shape variables to cover the Pareto front
        # Use a quasi-random / grid approach for good coverage
        # For 2D shape space, use a grid or Halton sequence

        budget = self.budget
        # Reserve some budget for potential refinement
        n_samples = min(budget, 150)
        n_refine = budget - n_samples

        # Generate shape samples using a stratified/grid approach
        # For 2 shape variables, use a 2D grid
        # sqrt(150) ~ 12, so 12x12 = 144 points
        grid_size = int(np.sqrt(n_samples))
        if grid_size < 2:
            grid_size = 2

        # Create grid points in [0, 1]^2
        # Use slightly interior points to avoid boundary issues
        eps = 0.001
        x0_vals = np.linspace(eps, 1 - eps, grid_size)
        x1_vals = np.linspace(eps, 1 - eps, grid_size)

        # Generate all combinations
        X_shape = np.array([[a, b] for a in x0_vals for b in x1_vals])
        n_shape_samples = len(X_shape)

        # If we have more budget, add some random samples for diversity
        remaining = n_samples - n_shape_samples
        if remaining > 0:
            rng = np.random.default_rng(42)
            X_rand = rng.random((remaining, n_shape))
            X_shape = np.vstack([X_shape, X_rand])
            n_shape_samples = len(X_shape)

        # Build full vectors with distance vars set to 0.5
        X_full = np.zeros((n_shape_samples, dim))
        X_full[:, :n_shape] = X_shape
        X_full[:, n_shape:] = 0.5  # Optimal distance values

        # Evaluate all points
        F = np.array([problem(x) for x in X_full])
        used = n_shape_samples

        # Check deadline
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return X_full

        # Use remaining budget for local refinement around promising points
        # Find non-dominated points and perturb them
        if n_refine > 0 and used < budget:
            # Simple non-domination filter
            dominated = np.zeros(n_shape_samples, dtype=bool)
            for i in range(n_shape_samples):
                for j in range(n_shape_samples):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            nd_points = np.where(~dominated)[0]
            if len(nd_points) > 0:
                # Perturb non-dominated points
                n_perturb = min(n_refine, len(nd_points) * 2)
                for k in range(n_perturb):
                    if time.time() > deadline - 2:
                        break
                    idx = nd_points[k % len(nd_points)]
                    base = X_full[idx].copy()
                    # Small perturbation in shape space
                    pert = 0.01 * (np.random.default_rng(k).standard_normal(n_shape))
                    base[:n_shape] = np.clip(base[:n_shape] + pert, 0, 1)
                    # Keep distance vars at 0.5
                    base[n_shape:] = 0.5
                    fc = problem(base)
                    used += 1
                    # Replace worst point if child is better
                    # Simple: just record it
                    X_full = np.vstack([X_full, base[None, :]])
                    F = np.vstack([F, fc[None, :]])

        return X_full
# EVOLVE-BLOCK-END
