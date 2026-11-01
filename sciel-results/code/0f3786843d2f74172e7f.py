import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    def fcc_lattice_init(n=100, box_size=1.0):
        """
        Generate initial points based on an FCC lattice, jittered and clipped.
        FCC is denser than simple cubic.
        """
        rng = np.random.default_rng(42)
        
        # Create a grid of FCC points
        # FCC unit cell has points at (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        # We need enough points to cover the volume and then pick/subsample or optimize.
        # Side length 1.
        # Let's create a grid of potential centers.
        # Approximate number of points in FCC per unit volume is 2 / (a^3) where a is lattice constant.
        # For r ~ 0.11, d ~ 0.22. a ~ 0.22 / sqrt(2) ~ 0.155.
        # Number of cells ~ (1/0.155)^3 ~ 270.
        # Let's just generate a dense grid and jitter it.
        
        nx = 12
        ny = 12
        nz = 12
        step = 1.0 / nx
        
        points = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = i * step + 0.5 * step
                    y = j * step + 0.5 * step
                    z = k * step + 0.5 * step
                    # Add FCC offsets
                    points.append([x, y, z])
                    points.append([x + 0.5*step, y + 0.5*step, z])
                    points.append([x + 0.5*step, y, z + 0.5*step])
                    points.append([x, y + 0.5*step, z + 0.5*step])
        
        points = np.array(points)
        # Clip to box
        points = np.clip(points, 0.0, 1.0)
        
        # We have too many points. We need to select N or just use a subset for init.
        # Actually, for optimization, we just need N points.
        # Let's pick N random points from this dense cloud? No, that's just random.
        # Better: Use a grid of N points.
        
        # Let's go back to a structured grid but with FCC-like symmetry if possible,
        # or just a well-spaced grid.
        # A 5x5x4 grid is 100 points.
        # Let's use a 5x5x4 grid, but map it to a slightly distorted space to break symmetry.
        
        nx, ny, nz = 5, 5, 4
        xs = np.linspace(0.1, 0.9, nx)
        ys = np.linspace(0.1, 0.9, ny)
        zs = np.linspace(0.1, 0.9, nz)
        
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        
        # Jitter
        grid += rng.uniform(-0.05, 0.05, size=(N, 3))
        grid = np.clip(grid, 0.0, 1.0)
        
        # Center in [-0.5, 0.5]
        grid -= 0.5
        
        return grid

    def energy_and_grad(x_flat, r_target):
        """
        Calculate energy and gradient for L-BFGS-B.
        Energy: Sum of squared violations for pairs and walls.
        """
        pts = x_flat.reshape(N, 3)
        
        # Pairwise distances
        # To avoid O(N^2) memory in gradient calculation if N is large, 
        # but N=100 is small enough.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Violations
        min_dist = 2.0 * r_target
        # d < min_dist -> violation
        viol_mask = dists < min_dist
        viol_dists = np.where(viol_mask, dists, 0.0)
        
        # Energy
        pair_viol = np.maximum(min_dist - dists, 0.0)
        pair_energy = np.sum(pair_viol**2)
        
        # Walls
        # Distance to wall: 0.5 - |coord|
        # Violation if r_target > 0.5 - |coord| => |coord| > 0.5 - r_target
        abs_pts = np.abs(pts)
        wall_limit = 0.5 - r_target
        wall_viol = np.maximum(abs_pts - wall_limit, 0.0)
        wall_energy = np.sum(wall_viol**2)
        
        total_energy = pair_energy + wall_energy
        
        # Gradient
        # d/dx_i ( (min_dist - d_ij)^2 ) = 2 * (min_dist - d_ij) * (-1) * d/dx_i d_ij
        # d/dx_i d_ij = (x_i - x_j) / d_ij
        # So grad_i = sum_j 2 * (min_dist - d_ij) * (x_i - x_j) / d_ij  (for d_ij < min_dist)
        
        grad = np.zeros_like(pts)
        
        # Only compute for violating pairs to save time and avoid div by zero
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < min_dist:
                    diff_vec = pts[i] - pts[j]
                    # Factor: 2 * (min_dist - d) / d
                    factor = 2.0 * (min_dist - d) / d
                    grad[i] += factor * diff_vec
                    grad[j] -= factor * diff_vec
        
        # Wall Gradient
        # d/dx_i ( (|x_i| - wall_limit)^2 ) = 2 * (|x_i| - wall_limit) * sign(x_i)
        for i in range(N):
            for k in range(3):
                val = abs_pts[i, k]
                limit = wall_limit
                if val > limit:
                    # d/dx ( (|x| - L)^2 ) = 2(|x|-L) * sign(x)
                    grad[i, k] += 2.0 * (val - limit) * np.sign(pts[i, k])
        
        return total_energy, grad.flatten()

    def compute_radius(pts):
        """Compute the valid radius for a configuration."""
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist, min_wall_dist)

    def optimize_for_radius(r_target, init_pts, time_limit=10.0):
        """
        Run L-BFGS-B to minimize violations for a given r_target.
        """
        start_time = time.time()
        
        x0 = init_pts.flatten()
        
        # L-BFGS-B options
        # maxiter is important.
        # We don't have a callback to check time easily in standard minimize,
        # but we can set maxiter high and rely on convergence or failure.
        # Better: use a wrapper or just set a reasonable maxiter.
        
        res = minimize(
            lambda x: energy_and_grad(x, r_target)[0],
            x0,
            jac=lambda x: energy_and_grad(x, r_target)[1],
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        return res.x.reshape(N, 3), res.fun

    def binary_search_radius():
        lo = 0.10
        hi = 0.115
        best_radius = 0.0
        best_pts = None
        
        total_time_limit = 110.0
        start_total = time.time()
        
        # Initial points
        init_pts = fcc_lattice_init()
        
        # Coarse search
        for i in range(12):
            if time.time() - start_total > total_time_limit:
                break
                
            mid = (lo + hi) / 2.0
            
            # Use the previous best points as init if available, else grid
            if best_pts is not None:
                curr_init = best_pts
            else:
                curr_init = init_pts
            
            pts_candidate, energy = optimize_for_radius(mid, curr_init)
            
            # Check validity
            # Energy should be close to 0
            if energy < 1e-6:
                r_achieved = compute_radius(pts_candidate)
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_pts = pts_candidate
                lo = mid
            else:
                hi = mid
        
        if best_pts is None:
            # Fallback
            best_pts = init_pts
            best_radius = compute_radius(best_pts)
            
        return best_pts

    return binary_search_radius()
# EVOLVE-BLOCK-END
