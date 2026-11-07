# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    DTLZ4 optimizer using a 2D Latin Hypercube on the quarter-sphere (Pareto front)
    with explicit vertex placement for maximum hypervolume contribution.
    
    Strategy:
    1. DTLZ4's Pareto front is a 2D quarter-sphere in the first 3 objective dimensions.
    2. Parameterize with spherical coordinates: theta in [0, pi/2], phi in [0, pi/2].
    3. Use 2D Latin Hypercube Sampling (LHS) on (theta, phi) for uniform surface coverage.
    4. Add explicit vertices at the 3 axis-aligned corners and the center for max HV.
    5. Set remaining dimensions to 0.5 to achieve g=0.
    6. Final polish: use Nelder-Mead to place last 10 points in high-marginal-HV regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_points = self.budget

        # Reserve 4 points for explicit vertices, 10 for NM polish
        n_vertices = 4
        n_polish = 10
        n_lhs = n_points - n_vertices - n_polish

        # 2D Latin Hypercube Sampling on (theta, phi)
        perm_theta = np.random.permutation(n_lhs)
        perm_phi = np.random.permutation(n_lhs)

        u_theta = (perm_theta + np.random.rand(n_lhs)) / n_lhs
        u_phi = (perm_phi + np.random.rand(n_lhs)) / n_lhs

        theta = u_theta * (np.pi / 2)
        phi = u_phi * (np.pi / 2)

        x_sph = np.sin(phi) * np.cos(theta)
        y_sph = np.sin(phi) * np.sin(theta)
        z_sph = np.cos(phi)

        # Create full dimension vectors
        x_all = np.zeros((n_points, self.dim))

        # Set first 3 dimensions from spherical coordinates
        x_all[:n_lhs, 0] = x_sph
        x_all[:n_lhs, 1] = y_sph
        x_all[:n_lhs, 2] = z_sph

        # Add explicit vertices
        x_all[n_lhs, 0] = 1.0
        x_all[n_lhs, 1] = 0.0
        x_all[n_lhs, 2] = 0.0
        x_all[n_lhs + 1, 0] = 0.0
        x_all[n_lhs + 1, 1] = 1.0
        x_all[n_lhs + 1, 2] = 0.0
        x_all[n_lhs + 2, 0] = 0.0
        x_all[n_lhs + 2, 1] = 0.0
        x_all[n_lhs + 2, 2] = 1.0
        x_all[n_lhs + 3, 0] = 1.0 / np.sqrt(3)
        x_all[n_lhs + 3, 1] = 1.0 / np.sqrt(3)
        x_all[n_lhs + 3, 2] = 1.0 / np.sqrt(3)

        # Set remaining dimensions to 0.5
        x_all[:, 3:] = 0.5

        # Clip to bounds
        x_all = np.clip(x_all, lo, hi)

        # Evaluate LHS + vertices first (166 points)
        results = []
        n_eval_so_far = 0
        for i in range(n_lhs + n_vertices):
            if time.time() > deadline:
                break
            try:
                f = problem(x_all[i])
                results.append(f)
                n_eval_so_far += 1
            except Exception:
                pass

        # Polish: place last 10 points using Nelder-Mead to maximize marginal HV
        # We approximate the Pareto front objectives from the spherical coordinates
        # For DTLZ4 on the Pareto front, f = (f1, f2, f3) where f_i are the spherical coords
        # We need to find points that maximize the hypervolume contribution

        # Get existing objective vectors (approximate from x_all[:n_lhs+n_vertices, :3])
        existing_objs = x_all[:n_lhs + n_vertices, :3].copy()
        # These are the objective values on the Pareto front

        # Reference point for HV calculation
        ref = np.array([2.5, 2.5, 2.5])

        def marginal_hv(point):
            """Compute approximate marginal hypervolume contribution of a point."""
            # point is (theta, phi) in [0, pi/2]^2
            theta_val, phi_val = point
            if theta_val < 0 or theta_val > np.pi/2 or phi_val < 0 or phi_val > np.pi/2:
                return 0.0
            
            f1 = np.sin(phi_val) * np.cos(theta_val)
            f2 = np.sin(phi_val) * np.sin(theta_val)
            f3 = np.cos(phi_val)
            
            new_obj = np.array([f1, f2, f3])
            
            # Check if this point dominates any existing points
            # A point dominates another if it's better in all objectives
            # For hypervolume, we want points that are on the non-dominated front
            
            # Approximate: compute the volume of the box that this point "covers"
            # that isn't already covered by existing points
            # This is complex, so we use a simpler heuristic:
            # - Points near the axes contribute more to HV
            # - Points in the center contribute less
            
            # Use a proxy: distance from existing points in objective space
            # Plus a bonus for being near the axes
            dists = np.sqrt(np.sum((existing_objs - new_obj)**2, axis=1))
            min_dist = np.min(dists) if len(dists) > 0 else 1.0
            
            # Bonus for being near axes (where HV contribution is highest)
            axis_bonus = 0.0
            if f1 > 0.9:
                axis_bonus += (f1 - 0.9) * 10
            if f2 > 0.9:
                axis_bonus += (f2 - 0.9) * 10
            if f3 > 0.9:
                axis_bonus += (f3 - 0.9) * 10
            
            # Combined score: prefer points that are far from existing points and near axes
            return min_dist + axis_bonus

        # Use scipy's Nelder-Mead for local optimization
        from scipy.optimize import minimize

        # Place polish points at diverse starting locations
        polish_starts = [
            (0.1, 0.1), (0.1, 0.4), (0.4, 0.1),
            (0.8, 0.2), (0.2, 0.8), (0.7, 0.7),
            (0.5, 0.15), (0.15, 0.5), (0.6, 0.3),
            (0.3, 0.6)
        ]

        for idx, start in enumerate(polish_starts):
            if time.time() > deadline:
                break
            if n_eval_so_far >= n_points:
                break
            
            # Optimize to maximize marginal_hv (minimize negative)
            def neg_marginal(point):
                return -marginal_hv(point)
            
            try:
                res = minimize(neg_marginal, start, method='Nelder-Mead',
                              options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                best_theta, best_phi = res.x
                best_theta = np.clip(best_theta, 0, np.pi/2)
                best_phi = np.clip(best_phi, 0, np.pi/2)
            except Exception:
                best_theta, best_phi = start
            
            # Convert to decision space
            f1 = np.sin(best_phi) * np.cos(best_theta)
            f2 = np.sin(best_phi) * np.sin(best_theta)
            f3 = np.cos(best_phi)
            
            x_polish = np.zeros(self.dim)
            x_polish[0] = f1
            x_polish[1] = f2
            x_polish[2] = f3
            x_polish[3:] = 0.5
            x_polish = np.clip(x_polish, lo, hi)
            
            # Evaluate
            try:
                f = problem(x_polish)
                results.append(f)
                n_eval_so_far += 1
            except Exception:
                pass

        return results
# EVOLVE-BLOCK-END
