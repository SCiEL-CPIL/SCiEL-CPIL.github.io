"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import itertools


def construct():
    """
    Use a multi-start local optimization approach to pack N equal circles
    inside the unit circle.
    
    Strategy:
    1. Generate initial configurations using various structured patterns
       (concentric rings, spiral, etc.)
    2. For each configuration, run a local optimization that maximizes the
       minimum distance between centers and to the boundary.
    3. Return the best configuration found.
    """
    
    def min_radius(x):
        """Compute the minimum radius for a given set of centers."""
        x = x.reshape(-1, 2)
        # Distance to boundary
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        # Pairwise distances
        n = len(x)
        if n <= 1:
            return float(np.min(dist_boundary))
        
        # Compute all pairwise distances
        # Use vectorized computation for speed
        diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        return float(np.min([np.min(dist_boundary), min_pair_dist / 2.0]))
    
    def negative_min_radius(x):
        """Negative of min_radius for minimization."""
        return -min_radius(x)
    
    def run_local_opt(x0, max_iter=500):
        """Run local optimization from initial point x0."""
        try:
            # Bounds: each coordinate in [-1, 1]
            bounds = Bounds([-1.0, -1.0] * N, [1.0, 1.0] * N)
            
            # Use L-BFGS-B for optimization
            result = minimize(
                negative_min_radius,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -0.05:
                return result.x, -result.fun
            else:
                return x0, min_radius(x0)
        except Exception as e:
            return x0, min_radius(x0)
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Concentric rings
    for n_rings in range(3, 15):
        if n_rings > 1:
            radii = np.linspace(0.05, 0.95, n_rings)
            points = []
            for i, r in enumerate(radii):
                if i == 0:
                    # Center point
                    if len(points) < N:
                        points.append([0.0, 0.0])
                else:
                    # Number of points on this ring
                    # Circumference / approximate spacing
                    n_pts = max(1, int(2 * np.pi * r / 0.1))
                    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                    for a in angles:
                        if len(points) < N:
                            points.append([r * np.cos(a), r * np.sin(a)])
            if len(points) >= N:
                initial_configs.append(np.array(points[:N]).flatten())
    
    # 2. Spiral (Archimedean)
    theta = np.linspace(0, 20 * np.pi, N)
    r = 0.95 * theta / (20 * np.pi)
    x_spiral = r * np.cos(theta)
    y_spiral = r * np.sin(theta)
    initial_configs.append(np.column_stack([x_spiral, y_spiral]).flatten())
    
    # 3. Random with jitter
    rng = np.random.RandomState(42)
    for seed in range(10):
        rng = np.random.RandomState(seed)
        # Generate random points in unit disk
        while True:
            r = np.sqrt(rng.random(N))
            theta = rng.random(N) * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Check if roughly valid
            if np.all(np.hypot(x, y) <= 0.98):
                break
        initial_configs.append(np.column_stack([x, y]).flatten())
    
    # 4. Grid-based
    for spacing in [0.06, 0.07, 0.08, 0.09]:
        pts = []
        for i in range(-20, 21):
            for j in range(-20, 21):
                x = i * spacing
                y = j * spacing * np.sqrt(3) / 2 + (i % 2) * spacing / 2
                if np.hypot(x, y) <= 0.98:
                    pts.append([x, y])
        if len(pts) >= N:
            initial_configs.append(np.array(pts[:N]).flatten())
    
    # Run local optimization on each initial config
    best_x = None
    best_r = 0.0
    
    for i, x0 in enumerate(initial_configs):
        if len(x0) != 2 * N:
            continue
        x_opt, r_opt = run_local_opt(x0, max_iter=300)
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt
    
    # If we didn't find a good solution, use a fallback
    if best_x is None or best_r < 0.03:
        # Fallback: use a simple ring-based approach
        n_rings = 10
        radii = np.linspace(0.05, 0.95, n_rings)
        points = []
        for i, r in enumerate(radii):
            if i == 0:
                points.append([0.0, 0.0])
            else:
                n_pts = max(1, int(2 * np.pi * r / 0.08))
                angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                for a in angles:
                    if len(points) < N:
                        points.append([r * np.cos(a), r * np.sin(a)])
        best_x = np.array(points[:N]).flatten()
        best_r = min_radius(best_x)
    
    # Final refinement with more iterations
    best_x, best_r = run_local_opt(best_x, max_iter=1000)
    
    return best_x.reshape(-1, 2)
# EVOLVE-BLOCK-END
