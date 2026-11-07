"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytic Pareto front sampler for DTLZ2 using stratified grid in decision space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 Analysis:
        # To reach the Pareto front, we must minimize g(x).
        # For standard DTLZ2, g(x) = sum((x_i - 0.5)^2) for i > m.
        # Minimum g=0 is achieved when x_i = 0.5 for i > m.
        # The Pareto front is the unit sphere in the positive orthant.
        
        # We parameterize the Pareto front using the first m = n_obj - 1 decision variables.
        # For n_obj=3, m=2.
        # x_0, x_1 in [0, 1] map to the quarter sphere.
        # f_1 = sin(x_0 * pi / 2)
        # f_2 = cos(x_0 * pi / 2) * sin(x_1 * pi / 2)
        # f_3 = cos(x_0 * pi / 2) * cos(x_1 * pi / 2)
        
        # To maximize hypervolume, we want a set of points that uniformly covers the Pareto front.
        # The "surface area measure" of the sphere is not uniform in the angle space (theta1, theta2)
        # if we just pick uniform angles? Actually, uniform angles in spherical coordinates
        # do NOT give uniform surface area density. The surface area element is sin(theta1) dtheta1 dtheta2.
        # So uniform sampling in theta1, theta2 clusters points near the poles (theta1=0, pi/2).
        
        # The decision variables x_0, x_1 are linearly related to the angles:
        # theta_1 = x_0 * pi / 2
        # theta_2 = x_1 * pi / 2
        # So uniform sampling in x_0, x_1 is equivalent to uniform sampling in theta_1, theta_2.
        # This clustering might hurt hypervolume if the "corners" of the Pareto front are less important
        # or if the density is too high there.
        
        # However, the instruction suggests: "mapped via the inverse CDF of the Pareto front's surface area measure".
        # If we want uniform surface area density, we should sample such that the number of points
        # in a patch is proportional to its surface area.
        # Surface area element dA = sin(theta1) dtheta1 dtheta2.
        # If we sample u1, u2 uniformly in [0,1], and set:
        # theta2 = u2 * pi / 2  (uniform in theta2 is correct for cylindrical symmetry)
        # theta1: we need dP = sin(theta1) dtheta1 / 2 (normalized).
        # CDF for theta1: P(theta1 < t) = (1 - cos(t)) / 2.
        # Inverse CDF: u1 = (1 - cos(theta1)) / 2 => cos(theta1) = 1 - 2*u1 => theta1 = arccos(1 - 2*u1).
        
        # Let's test if uniform x0, x1 (current) or inverse CDF mapping is better.
        # The current program uses Sobol in angle space (uniform angles).
        # The inspiration mentions "uniform grid in x0, x1 space".
        # Let's try a stratified grid in the variable space that accounts for the Jacobian.
        # Actually, a simple uniform grid in x0, x1 is often very effective for DTLZ2 because the mapping is smooth.
        # But to be "structurally different" and follow the hint about surface area measure:
        
        # Let's generate a grid of points in a transformed space that yields uniform surface area density.
        # Let u1, u2 be uniform in [0, 1].
        # theta2 = u2 * pi / 2
        # theta1 = arccos(1 - 2 * u1)
        
        # Then x_0 = 2 * theta1 / pi
        # x_1 = 2 * theta2 / pi
        
        # We need to generate `budget` points.
        # A grid is efficient. Let's choose grid dimensions.
        # For 2D, a square-ish grid is good.
        # n_points = budget
        # Let's find n_grid such that n_grid^2 >= budget.
        # Or simply n_grid = ceil(sqrt(budget)).
        
        n_grid = int(np.ceil(np.sqrt(self.budget)))
        # We might have more points than budget, so we'll take the first `budget` from the grid.
        # Or we can adjust n_grid to be closer to budget.
        
        # Let's generate a meshgrid of u1, u2 in [0, 1]
        # Using linspace with endpoint=True? Or just random?
        # Stratified grid:
        # u1_vals = (np.arange(n_grid) + 0.5) / n_grid  # Midpoint of each cell
        # u2_vals = (np.arange(n_grid) + 0.5) / n_grid
        
        # This creates n_grid^2 points.
        # We flatten and take the first `budget` points.
        # However, taking a prefix of a flattened grid might be biased if the grid is ordered in a specific way.
        # It's better to generate exactly `budget` points if possible, or take a random subset?
        # No, deterministic is better for reproducibility and "low discrepancy".
        
        # Let's try to fill the grid as much as possible.
        # If budget is not a perfect square, we can just use the first `budget` points from the flattened grid.
        # The order of flattening matters. Row-major is standard.
        
        u1_vals = (np.arange(n_grid) + 0.5) / n_grid
        u2_vals = (np.arange(n_grid) + 0.5) / n_grid
        
        U1, U2 = np.meshgrid(u1_vals, u2_vals, indexing='ij')
        
        # Flatten
        U1_flat = U1.flatten()
        U2_flat = U2.flatten()
        
        # Take first `budget` points
        U1_flat = U1_flat[:self.budget]
        U2_flat = U2_flat[:self.budget]
        
        # Map to theta
        theta2 = U2_flat * np.pi / 2
        # Inverse CDF for uniform surface area density on the sphere cap
        # P(theta1 < t) = (1 - cos(t)) / 2
        # u1 = (1 - cos(theta1)) / 2
        # cos(theta1) = 1 - 2 * u1
        theta1 = np.arccos(1 - 2 * U1_flat)
        
        # Map to x
        x1 = 2 * theta1 / np.pi
        x2 = 2 * theta2 / np.pi
        
        # Construct the x vectors
        x_vectors = np.zeros((self.budget, self.dim))
        x_vectors[:, 0] = x1
        x_vectors[:, 1] = x2
        x_vectors[:, 2:] = 0.5
        
        # Clip to bounds
        x_vectors = np.clip(x_vectors, lo, hi)
        
        # Evaluate all points
        results = []
        for i in range(self.budget):
            if time.time() > deadline:
                break
            x = x_vectors[i]
            f = problem(x)
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END
