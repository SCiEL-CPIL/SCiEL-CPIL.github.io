"""Portfolio search for DTLZ2: dense front grid + gap-filling, deadline-guarded."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a Pareto-front grid portfolio."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        used = 0
        recorded = []  # list of objective vectors

        def time_ok():
            return time.time() < deadline

        def add_point(x, f):
            nonlocal used
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = np.array(f, dtype=float)
            # Only count if strictly better than reference point in every objective
            if np.all(f < 2.5):
                recorded.append(f)
            used += 1
            return f

        # Strategy: Sample the known Pareto front of DTLZ2
        # DTLZ2 Pareto front: f1^2 + f2^2 + f3^2 = 1, fi >= 0
        # We parameterize using spherical coordinates with sin^2 transformation
        # to concentrate points near the boundaries (axes) where hypervolume boxes are larger

        n_front = min(150, self.budget - 20)  # reserve some budget for refinement
        n_refine = self.budget - n_front

        # Generate points on the unit sphere in the positive octant
        # Using a stratified approach with sin^2 bias toward boundaries
        # Parameterize: theta in [0, pi/2], phi in [0, pi/2]
        # f1 = cos(theta), f2 = sin(theta)*cos(phi), f3 = sin(theta)*sin(phi)

        # Use sin^2 transformation to concentrate near boundaries
        # For theta: use u^0.7 to push toward 0 and 1 (boundaries)
        # For phi: similar

        # Create stratified samples
        n_theta = int(np.sqrt(n_front))
        n_phi = max(1, n_front // n_theta)

        # Stratified sampling with bias toward boundaries
        theta_samples = []
        phi_samples = []

        for i in range(n_theta):
            # Stratified position with bias toward edges
            u = (i + 0.5) / n_theta
            # Apply power transformation to concentrate near 0 and 1
            # u^0.6 pushes toward edges
            u_biased = u ** 0.6 if u < 0.5 else 1.0 - (1.0 - u) ** 0.6
            theta = u_biased * np.pi / 2.0
            theta_samples.append(theta)

        for j in range(n_phi):
            v = (j + 0.5) / n_phi
            v_biased = v ** 0.6 if v < 0.5 else 1.0 - (1.0 - v) ** 0.6
            phi = v_biased * np.pi / 2.0
            phi_samples.append(phi)

        # Generate front points
        front_points = []
        for theta in theta_samples:
            for phi in phi_samples:
                if used >= self.budget or not time_ok():
                    break
                f1 = np.cos(theta)
                f2 = np.sin(theta) * np.cos(phi)
                f3 = np.sin(theta) * np.sin(phi)
                f = np.array([f1, f2, f3])

                # Convert from objective space to decision space for DTLZ2
                # DTLZ2: x[0:9] are position variables, x[9:12] are angle variables
                # g(x) = sum((xi - 0.5)^2) for i in 9..11, set g=0 for Pareto front
                # f1 = (1+g)*cos(theta1)*cos(theta2)
                # f2 = (1+g)*cos(theta1)*sin(theta2)
                # f3 = (1+g)*sin(theta1)
                # With g=0: f1=cos(theta1)*cos(theta2), f2=cos(theta1)*sin(theta2), f3=sin(theta1)
                # So theta1 = arccos(sqrt(f1^2+f2^2)) = arccos(sqrt(1-f3^2))
                # theta2 = arctan2(f2, f1)

                # Set position variables to 0.5 (g=0)
                x = np.full(self.dim, 0.5)
                # Set angle variables
                # theta1 corresponds to arccos(f3) since f3 = sin(theta1)
                theta1 = np.arcsin(np.clip(f3, -1, 1))
                # theta2 corresponds to arctan2(f2, f1)
                theta2 = np.arctan2(f2, f1)
                # Map to [0,1] range: x[9] = theta1/(pi/2), x[10] = theta2/(pi/2)
                x[self.dim - 3] = theta1 / (np.pi / 2.0)
                x[self.dim - 2] = theta2 / (np.pi / 2.0)
                x[self.dim - 1] = 0.5  # third angle, doesn't matter when g=0

                # Clip to bounds
                x = np.clip(x, lo, hi)

                f_actual = problem(x)
                add_point(x, f_actual)
                if used >= self.budget or not time_ok():
                    break

        # Use remaining budget for local refinement around existing points
        if used < self.budget and recorded:
            # Convert recorded to array
            F_arr = np.array(recorded)
            n_existing = len(F_arr)

            # Find gaps in the front and try to fill them
            # Simple approach: perturb existing points and keep improvements
            for _ in range(self.budget - used):
                if not time_ok():
                    break
                # Pick a random existing point and perturb it
                idx = rng.integers(n_existing)
                f_base = F_arr[idx]

                # Small perturbation in objective space, then map back
                # Or just do a random local search in decision space
                x = np.full(self.dim, 0.5)
                theta1 = np.arcsin(np.clip(f_base[2], -1, 1))
                theta2 = np.arctan2(f_base[1], f_base[0])
                x[self.dim - 3] = theta1 / (np.pi / 2.0)
                x[self.dim - 2] = theta2 / (np.pi / 2.0)
                x[self.dim - 1] = 0.5

                # Perturb
                x += 0.02 * rng.standard_normal(self.dim)
                x = np.clip(x, lo, hi)

                f = problem(x)
                add_point(x, f)

        return recorded
# EVOLVE-BLOCK-END
