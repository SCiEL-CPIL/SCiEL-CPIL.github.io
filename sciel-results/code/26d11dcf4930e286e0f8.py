"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares to maximize the minimum radius.
    Formulation:
    Variables: x (2N coords), r (radius)
    Residuals:
    - For each pair (i,j): 2*r - dist(i,j)
    - For each point i: r - (1 - norm(c_i))
    Minimization of sum of squares pushes r up until constraints are tight.
    """
    n = N
    
    def get_hexagonal_points(n_points, radius=0.95):
        """Generate a hexagonal lattice of points inside a circle of given radius."""
        # Estimate spacing
        # Area of circle ~ pi * radius^2. Area per point ~ 2 * (sqrt(3)/2) * s^2 = sqrt(3) * s^2
        # s ~ sqrt(pi * radius^2 / (n_points * sqrt(3)))
        s = np.sqrt(np.pi * radius**2 / (n_points * np.sqrt(3)))
        
        points = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= radius**2:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n_points:
            # Fallback: increase density or use grid
            # Try smaller s
            s *= 0.9
            points = []
            k = int(1.0 / s) + 2
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if x**2 + y**2 <= radius**2:
                        points.append([x, y])
            points = np.array(points)
            if len(points) < n_points:
                # Just take a grid
                xs = np.linspace(-radius, radius, 20)
                ys = np.linspace(-radius, radius, 20)
                grid = np.array([[x, y] for x in xs for y in ys])
                points = grid[:n_points]
        else:
            # Select n_points closest to origin or random?
            # Closest to origin is usually better for symmetry
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            points = points[idx[:n_points]]
            
        return points

    def compute_radius(centers):
        """Compute the packing radius for given centers."""
        if len(centers) < 2:
            return 0.0
        dists = pdist(centers)
        min_pair_dist = np.min(dists)
        boundary_dists = 1.0 - np.linalg.norm(centers, axis=1)
        min_boundary_dist = np.min(boundary_dists)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def residuals(x):
        """
        x: [x1, y1, ..., xn, yn, r]
        Returns residuals for least_squares.
        """
        centers = x[:2*n].reshape(n, 2)
        r = x[-1]
        
        # Pairwise distances
        dists = pdist(centers)
        # Residuals: 2*r - dist
        pair_res = 2 * r - dists
        
        # Boundary distances
        norms = np.linalg.norm(centers, axis=1)
        boundary_dists = 1.0 - norms
        # Residuals: r - boundary_dist
        bound_res = r - boundary_dists
        
        return np.concatenate([pair_res, bound_res])

    # Initial points
    initial_points = get_hexagonal_points(n, radius=0.9)
    r_init = compute_radius(initial_points)
    
    best_radius = 0.0
    best_centers = initial_points
    
    # Multi-restart strategy
    # We try a few times with different initializations/perturbations
    rng = np.random.default_rng(42)
    
    max_restarts = 5
    for i in range(max_restarts):
        if i == 0:
            centers_init = initial_points
        else:
            # Perturb the previous best or a new lattice
            if i % 2 == 1:
                centers_init = initial_points + rng.normal(0, 0.01, size=initial_points.shape)
            else:
                centers_init = best_centers + rng.normal(0, 0.01, size=best_centers.shape)
        
        # Clip to ensure they are roughly inside
        norms = np.linalg.norm(centers_init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            centers_init[mask] *= 0.95 / norms[mask]
        
        r_current = compute_radius(centers_init)
        # Start r slightly smaller than current estimate to allow growth
        r_start = max(0.01, r_current * 0.8)
        
        x0 = np.concatenate([centers_init.flatten(), [r_start]])
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=2000, 
                                   xtol=1e-8, ftol=1e-8, gtol=1e-8)
            centers_opt = result.x[:2*n].reshape(n, 2)
            r_opt = result.x[-1]
            
            # Verify validity
            r_check = compute_radius(centers_opt)
            if r_check > best_radius:
                best_radius = r_check
                best_centers = centers_opt
        except Exception:
            continue
            
    # Final check
    final_r = compute_radius(best_centers)
    if final_r <= 0 or not np.isfinite(final_r):
        return initial_points
    
    return best_centers
# EVOLVE-BLOCK-END
