"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def construct():
    """
    Construct centers of N=300 equal circles inside the unit disk.
    Uses a density-driven approach: initialize on hex lattice, then optimize
    to maximize the minimum distance between centers and to the boundary.
    """
    # Step 1: Initialize on hexagonal lattice
    def hex_lattice_init(n_points):
        # Estimate spacing
        area = np.pi
        hex_area = (np.sqrt(3) / 2) * (2.0 ** 2)  # area per hex cell with spacing 2
        # Actually, for hex packing, density is pi/(2*sqrt(3)) ~ 0.9069
        # So effective area per point ~ area / n_points
        # Spacing d such that hex cell area ~ d^2 * sqrt(3)/2
        # d^2 * sqrt(3)/2 ~ pi/n_points => d ~ sqrt(2*pi/(n_points*sqrt(3)))
        d = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
        
        # Generate hex lattice points
        k = int(1.0 / d) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        # If we have more than n_points, take the closest to origin
        if len(points) > n_points:
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:n_points]
            points = points[idx]
        
        # If we have fewer, add random points near the boundary
        while len(points) < n_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.8, 0.95)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
        
        return points[:n_points]
    
    # Step 2: Optimization using gradient ascent on minimum distance
    def objective(x):
        """Negative of the minimum radius (to minimize)."""
        x = x.reshape(-1, 2)
        # Distance to boundary
        dist_boundary = 1.0 - np.hypot(x[:, 0], x[:, 1])
        # Pairwise distances
        tree = cKDTree(x)
        # Get nearest neighbor for each point
        d, _ = tree.query(x, k=2)
        min_pair_dist = np.min(d[:, 1])
        
        # The radius is min(min_pair_dist/2, min_dist_boundary)
        r = min(min_pair_dist / 2.0, np.min(dist_boundary))
        
        # Add small penalty for points outside or too close to boundary
        penalty = 0.0
        outside = np.hypot(x[:, 0], x[:, 1]) > 1.0
        if np.any(outside):
            penalty += 10.0 * np.sum(np.hypot(x[outside, 0], x[outside, 1]) - 1.0)
        
        return -r + penalty
    
    def gradient(x):
        """Numerical gradient of the objective."""
        eps = 1e-5
        grad = np.zeros_like(x)
        flat = x.flatten()
        for i in range(len(flat)):
            flat_p = flat.copy()
            flat_m = flat.copy()
            flat_p[i] += eps
            flat_m[i] -= eps
            fp = objective(flat_p.reshape(-1, 2))
            fm = objective(flat_m.reshape(-1, 2))
            grad.flat[i] = (fp - fm) / (2 * eps)
        return grad
    
    # Initialize
    np.random.seed(42)
    x0 = hex_lattice_init(N)
    
    # Use Nelder-Mead for optimization (doesn't need gradient, more robust)
    result = minimize(
        objective,
        x0.flatten(),
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    x_opt = result.x.reshape(-1, 2)
    
    # Ensure all points are inside the unit disk
    radii = np.hypot(x_opt[:, 0], x_opt[:, 1])
    mask = radii <= 1.0
    if not np.all(mask):
        # Clip points that are outside
        for i in range(len(x_opt)):
            if radii[i] > 1.0:
                scale = 0.999 / radii[i]
                x_opt[i] *= scale
    
    return x_opt

# EVOLVE-BLOCK-END
