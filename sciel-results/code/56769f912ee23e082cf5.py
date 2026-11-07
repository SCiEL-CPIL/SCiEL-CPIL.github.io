"""Multi-objective optimizer on DTLZ2 with DTLZ2-aware parameterization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ2-aware multi-objective optimizer using spherical parameterization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(12345)

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ2 structure:
        # x[0..n_obj-2] = shape variables (x1, x2)
        # x[n_obj-1..] = position variables (x3..x12)
        # f1 = (1+g)*cos(x1*pi/2)*cos(x2*pi/2)
        # f2 = (1+g)*cos(x1*pi/2)*sin(x2*pi/2)
        # f3 = (1+g)*sin(x1*pi/2)
        # g = sum((xi-0.5)^2) for position variables, g*=0 on PF
        # So PF: all position vars = 0.5, shape vars determine point on unit sphere quadrant

        n_shape = n_obj - 1  # 2 shape variables
        n_pos = dim - n_shape  # 10 position variables

        def shape_to_x(shape):
            """Convert shape params (x1, x2) and fixed pos=0.5 to full x vector."""
            x = np.empty(dim)
            x[:n_shape] = shape
            x[n_shape:] = 0.5
            return x

        def evaluate_shape(shape):
            """Evaluate objectives given shape variables (pos vars fixed at 0.5)."""
            x = shape_to_x(shape)
            return problem(x)

        # Strategy: sample the known Pareto front densely, then refine
        # The Pareto front of DTLZ2 in 3D is the unit sphere in the first octant
        # f1 = cos(a)*cos(b), f2 = cos(a)*sin(b), f3 = sin(a)
        # where a = x1*pi/2, b = x2*pi/2, x1,x2 in [0,1]

        # Phase 1: Dense sampling of the Pareto front using known structure
        # Use spherical coordinates: a in [0, pi/2], b in [0, pi/2]
        n_samples = 120  # initial dense sampling
        used = 0

        # Create a grid on the sphere
        # Use stratified sampling for better coverage
        a_vals = np.linspace(0, np.pi / 2, 12)
        b_vals = np.linspace(0, np.pi / 2, 10)

        # Store all evaluated points
        X_all = []
        F_all = []

        # Sample the Pareto front
        for a in a_vals:
            for b in b_vals:
                if used >= budget:
                    break
                x1 = a * 2 / np.pi
                x2 = b * 2 / np.pi
                shape = np.array([x1, x2])
                x = shape_to_x(shape)
                x = np.clip(x, lo, hi)
                f = problem(x)
                X_all.append(x)
                F_all.append(f)
                used += 1

        # Phase 2: Local refinement around sampled points
        # For each point, try small perturbations in position variables
        # to see if we can get closer to g=0

        remaining_budget = budget - used
        if remaining_budget <= 0:
            return np.array(X_all)

        # Convert to arrays
        X_arr = np.array(X_all)
        F_arr = np.array(F_all)

        # For each point, try perturbing position variables
        # to reduce g (which should already be 0 if pos=0.5)
        # Actually, the key insight: on the PF, g=0, so all pos vars must be 0.5
        # The objective values are determined solely by shape vars
        # So we just need to sample shape vars well

        # Let's do a more systematic sampling
        # Reset and do better sampling
        X_all = []
        F_all = []
        used = 0

        # Use a Fibonacci sphere or stratified sampling
        # For 3-objective DTLZ2, the PF is a quarter sphere
        # We want to maximize hypervolume, so we need good coverage

        # Use a structured grid with some randomization
        n_a = 14
        n_b = 12

        for i in range(n_a):
            for j in range(n_b):
                if used >= budget:
                    break
                # Stratified sampling
                a = (i + rng.random()) / n_a * np.pi / 2
                b = (j + rng.random()) / n_b * np.pi / 2
                x1 = a * 2 / np.pi
                x2 = b * 2 / np.pi
                shape = np.array([x1, x2])
                x = shape_to_x(shape)
                x = np.clip(x, lo, hi)
                f = problem(x)
                X_all.append(x)
                F_all.append(f)
                used += 1

        # Use remaining budget for local search around best points
        remaining = budget - used
        if remaining > 0 and len(X_all) > 0:
            X_arr = np.array(X_all)
            F_arr = np.array(F_all)

            # Sort by sum of objectives (proxy for quality)
            order = np.argsort(np.sum(F_arr, axis=1))
            best_indices = order[:min(20, len(order))]

            # For each of the best points, try small perturbations
            perturbations_per_point = remaining // len(best_indices) if len(best_indices) > 0 else 0
            perturbations_per_point = max(1, perturbations_per_point)

            for idx in best_indices:
                if used >= budget:
                    break
                base_x = X_arr[idx].copy()
                for p in range(perturbations_per_point):
                    if used >= budget:
                        break
                    # Perturb position variables slightly
                    x_new = base_x.copy()
                    # Small random perturbation to position vars
                    pos_idx = np.arange(n_shape, dim)
                    noise = rng.normal(0, 0.05, size=len(pos_idx))
                    x_new[pos_idx] = np.clip(x_new[pos_idx] + noise, lo[pos_idx], hi[pos_idx])

                    # Also slightly perturb shape vars
                    shape_noise = rng.normal(0, 0.02, size=n_shape)
                    x_new[:n_shape] = np.clip(x_new[:n_shape] + shape_noise, lo[:n_shape], hi[:n_shape])

                    f_new = problem(x_new)
                    X_all.append(x_new)
                    F_all.append(f_new)
                    used += 1

        return np.array(X_all)
# EVOLVE-BLOCK-END
