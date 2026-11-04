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
    to maximize the local uniformity (Voronoi-like area proxy) using L-BFGS-B.
    """
    # Step 1: Initialize on hexagonal lattice
    def hex_lattice_init(n_points):
        d = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
        
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
        if len(points) > n_points:
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:n_points]
            points = points[idx]
        
        while len(points) < n_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.8, 0.95)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
        
        return points[:n_points]
    
    # Step 2: Objective function based on local uniformity (log of nearest neighbor distance)
    def objective(x):
        x = x.reshape(-1, 2)
        # Calculate nearest neighbor distances
        tree = cKDTree(x)
        d, _ = tree.query(x, k=2)
        nn_dists = d[:, 1]
        
        # Avoid log(0)
        nn_dists = np.maximum(nn_dists, 1e-10)
        
        # Maximize sum of log(nn_dist) -> Minimize negative sum
        val = -np.sum(np.log(nn_dists))
        
        # Boundary penalty
        radii = np.hypot(x[:, 0], x[:, 1])
        outside = np.maximum(0, radii - 1.0)
        penalty = 100.0 * np.sum(outside**2)
        
        return val + penalty

    # Step 3: Optimization
    np.random.seed(42)
    x0 = hex_lattice_init(N)
    
    # Use L-BFGS-B with numerical gradient
    # We don't use bounds because the constraint is circular, handled by penalty
    result = minimize(
        objective,
        x0.flatten(),
        method='L-BFGS-B',
        options={
            'maxiter': 1000,
            'ftol': 1e-10,
            'gtol': 1e-8,
            'maxls': 20,
            'maxcor': 20
        }
    )
    
    x_opt = result.x.reshape(-1, 2)
    
    # Step 4: Post-processing to ensure validity and maximize min distance
    # The log-sum objective maximizes local spacing, but we need to ensure the GLOBAL min distance is large.
    # We can do a few iterations of "push apart" on the closest pairs.
    
    for _ in range(50):
        # Find closest pair
        tree = cKDTree(x_opt)
        d, idx_pairs = tree.query(x_opt, k=2)
        min_d = np.min(d[:, 1])
        min_idx = np.argmin(d[:, 1])
        
        # If min distance is very small, push them apart
        # Also check boundary
        radii = np.hypot(x_opt[:, 0], x_opt[:, 1])
        
        # Simple heuristic: move the two closest points away from each other
        i, j = min_idx, idx_pairs[min_idx, 1]
        pi = x_opt[i]
        pj = x_opt[j]
        
        # Vector from j to i
        v = pi - pj
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-6:
            v = np.array([1.0, 0.0])
            norm_v = 1.0
        v = v / norm_v
        
        # Push apart by a small amount
        step = 0.001
        x_opt[i] += v * step
        x_opt[j] -= v * step
        
        # Re-constrain to circle
        for k in [i, j]:
            r_k = np.linalg.norm(x_opt[k])
            if r_k > 1.0:
                x_opt[k] *= (1.0 - 1e-6) / r_k

    # Final check and clip
    radii = np.hypot(x_opt[:, 0], x_opt[:, 1])
    mask = radii <= 1.0
    if not np.all(mask):
        for i in range(len(x_opt)):
            if radii[i] > 1.0:
                scale = 0.999 / radii[i]
                x_opt[i] *= scale
                
    return x_opt

# EVOLVE-BLOCK-END
