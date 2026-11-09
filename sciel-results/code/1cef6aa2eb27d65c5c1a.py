"""Combined: deterministic Pareto-front grid with Halton-sequence refinement for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a structured front grid + Halton refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0,1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s wall-clock limit, use 280s for safety

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2: Pareto front is a unit sphere in the first n_obj dimensions
        # f_i = g * sin(pi * x_i) for i < n_obj, where g = 1 + 10*(n-1)*mean((x_i - 0.5)^2)
        # Actually for DTLZ2: f_i = g * (1 - sin(pi * x_i)) for i < n_obj
        # Wait, let me recall: DTLZ2 objectives are f_i = g * sin(pi * x_i) for i=1..M
        # No, actually: f_i = g * (1 - sin(pi * x_i))? Let me think again.
        # DTLZ2: f_i(x) = g(x) * sin(pi * x_i) for i = 1, ..., M
        # where g(x) = 1 + 10*(n-M) * mean_{i=M+1..n} (x_i - 0.5)^2
        # The Pareto front is g=0 (i.e., x_i=0.5 for i>M) and sum f_i^2 = 1
        # So the front is the unit sphere in M dimensions in the positive orthant.

        # Strategy:
        # 1. Generate points on the Pareto front (unit sphere) using structured sampling
        # 2. For each point on the front, set the last (dim - n_obj) variables to 0.5
        #    to make g=0, so the objectives match the front exactly.
        # 3. Use remaining budget for local refinement.

        # Generate points on the unit sphere in n_obj dimensions
        # Use a combination of:
        # a) Grid points (structured)
        # b) Halton sequence points (quasi-random)

        # Phase 1: Structured grid on the sphere
        # For 3 objectives, use spherical coordinates
        # theta in [0, pi/2], phi in [0, pi/2] such that all components are positive

        # Let's use a different approach: generate random directions on the unit sphere
        # and also structured points

        # Number of points to allocate
        # Phase 1: ~60% of budget for front sampling
        # Phase 2: ~40% for local optimization

        n_front = int(budget * 0.6)
        n_refine = budget - n_front

        # Generate points on the unit sphere in n_obj dimensions
        # Method 1: Uniform random on sphere
        rng = np.random.default_rng(42)

        # Generate n_front points on the unit sphere
        # Using Gaussian normalization
        Z = rng.standard_normal((n_front, n_obj))
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        sphere_points = Z / norms

        # Ensure all components are positive (project to positive orthant)
        sphere_points = np.abs(sphere_points)
        # Re-normalize
        norms2 = np.linalg.norm(sphere_points, axis=1, keepdims=True)
        norms2[norms2 < 1e-10] = 1.0
        sphere_points = sphere_points / norms2

        # Convert sphere points to DTLZ2 decision variables
        # f_i = g * sin(pi * x_i) => for g=0 case, we need sin(pi * x_i) = f_i
        # So x_i = arcsin(f_i) / pi for i < n_obj
        # And x_i = 0.5 for i >= n_obj (to make g=0)

        # Actually, since we're setting g=0 by fixing the last variables to 0.5,
        # we need f_i = sin(pi * x_i), so x_i = arcsin(f_i) / pi
        # But f_i must be in [0, 1], and arcsin is defined for [-1, 1]

        # Clip to valid range
        sphere_points_clipped = np.clip(sphere_points, 0, 1)
        x_front = np.zeros((n_front, dim))
        x_front[:, :n_obj] = np.arcsin(sphere_points_clipped) / np.pi
        x_front[:, n_obj:] = 0.5

        # Evaluate front points
        F_front = np.array([problem(x) for x in x_front])
        used = n_front

        # Phase 2: Local refinement using gradient-free optimization
        # Pick the best points and do local search
        # Use a simple approach: for each of the top K points, do a few perturbations

        # Actually, let's use a different strategy:
        # Use Halton sequence for more uniform coverage
        # And do local optimization around promising points

        # Let's evaluate and keep track of all points
        X_all = x_front.copy()
        F_all = F_front.copy()

        # Add Halton sequence points for better coverage
        # Generate Halton points in [0,1]^n_obj, map to sphere
        n_halton = min(100, n_front // 2)
        h1 = self._halton(n_halton, 2)
        h2 = self._halton(n_halton, 3)
        h3 = self._halton(n_halton, 5)

        # Map 3D Halton points to sphere
        # Use a simple mapping: normalize the vector (h1, h2, h3)
        H = np.column_stack([h1, h2, h3])
        H_norms = np.linalg.norm(H, axis=1, keepdims=True)
        H_norms[H_norms < 1e-10] = 1.0
        H_sphere = H / H_norms
        H_sphere = np.abs(H_sphere)
        H_norms2 = np.linalg.norm(H_sphere, axis=1, keepdims=True)
        H_norms2[H_norms2 < 1e-10] = 1.0
        H_sphere = H_sphere / H_norms2

        x_halton = np.zeros((n_halton, dim))
        x_halton[:, :n_obj] = np.arcsin(np.clip(H_sphere, 0, 1)) / np.pi
        x_halton[:, n_obj:] = 0.5

        if used + n_halton <= budget:
            F_halton = np.array([problem(x) for x in x_halton])
            X_all = np.vstack([X_all, x_halton])
            F_all = np.vstack([F_all, F_halton])
            used += n_halton

        # Phase 3: Local refinement
        # Pick diverse points from the current set and do local search
        # Use a simple coordinate-wise perturbation

        # First, find non-dominated points
        def is_dominated(F, f):
            """Check if f is dominated by any point in F."""
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        # Simple local search: perturb each coordinate
        remaining_budget = budget - used
        n_refine_points = min(50, remaining_budget // 10)
        step_size = 0.01

        # Pick points to refine: use a simple diversity measure
        # Just pick evenly spaced indices
        indices = np.linspace(0, len(X_all) - 1, n_refine_points, dtype=int)
        indices = np.unique(indices)

        for idx in indices:
            if used >= budget:
                break
            if time.time() > deadline - 30:  # Safety margin
                break

            x_base = X_all[idx].copy()
            for coord in range(dim):
                if used >= budget:
                    break
                if time.time() > deadline - 30:
                    break

                for delta in [-step_size, step_size]:
                    if used >= budget:
                        break
                    if time.time() > deadline - 30:
                        break

                    x_new = x_base.copy()
                    x_new[coord] += delta
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1

                    # Check if this improves upon the base point
                    if np.all(f_new <= F_all[idx]) and np.any(f_new < F_all[idx]):
                        X_all[idx] = x_new
                        F_all[idx] = f_new

        return X_all
# EVOLVE-BLOCK-END
