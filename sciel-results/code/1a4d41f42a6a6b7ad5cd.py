# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0  # seconds to ensure we don't timeout

    def compute_ratio(points):
        """Compute the actual score: (d_min/d_max)^2"""
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(points_flat):
        """
        Smooth objective function for optimization.
        Uses a soft-min approximation to make the function differentiable.
        """
        points = points_flat.reshape(n, d)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 1.0
        # Use a smooth approximation: minimize -log(d_min) + log(d_max)
        # This is equivalent to minimizing -log(d_min/d_max)
        # Or simply: maximize d_min/d_max, so minimize -(d_min/d_max)
        r = d_min / d_max
        return -(r ** 2)

    def get_hexagonal_lattice_init():
        """Generate a hexagonal lattice initialization with 16 central points."""
        # Hexagonal lattice basis vectors
        a = 1.0  # lattice spacing
        b = a * np.sqrt(3)
        
        # Generate a larger hexagonal grid
        points = []
        # Use axial coordinates for hex grid
        for q in range(-3, 4):
            for r in range(-3, 4):
                # Convert axial to Cartesian
                x = a * (q + r/2)
                y = b * r/2
                points.append([x, y])
        
        points = np.array(points)
        
        # Sort by distance from origin and take the 16 closest
        dists_from_origin = np.sqrt(np.sum(points**2, axis=1))
        sorted_indices = np.argsort(dists_from_origin)
        hex_points = points[sorted_indices[:n]]
        
        # Center the points
        hex_points -= np.mean(hex_points, axis=0)
        
        return hex_points

    def get_two_ring_init():
        """Generate two concentric rings: 6 inner, 10 outer (or 8+8)."""
        # Inner ring: 6 points
        r_inner = 0.5
        angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
        p_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        
        # Outer ring: 10 points
        r_outer = 1.0
        angles_outer = np.linspace(np.pi/10, 2*np.pi + np.pi/10, 10, endpoint=False)
        p_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        
        return np.vstack([p_inner, p_outer])

    def get_regular_polygon_init():
        """Regular 16-gon."""
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def get_grid_init():
        """4x4 grid."""
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    best_points = None
    best_score = -1.0

    # Generate initializations - hexagonal first as it's closest to optimal
    initializations = [
        get_hexagonal_lattice_init(),
        get_two_ring_init(),
        get_regular_polygon_init(),
        get_grid_init()
    ]

    # First, evaluate all initializations
    for init_points in initializations:
        score = compute_ratio(init_points)
        if score > best_score:
            best_score = score
            best_points = init_points.copy()

    # Time budget for optimization
    opt_budget = deadline - 60.0  # Leave 60s for safety

    # Use Powell's method for each initialization
    for i, init_points in enumerate(initializations):
        if time.time() > start_time + opt_budget:
            break

        remaining_time = start_time + opt_budget - time.time()
        if remaining_time < 10:
            break

        try:
            x0 = init_points.flatten()
            
            # Powell's method is robust for non-convex problems
            res = minimize(
                smooth_objective,
                x0,
                method='Powell',
                options={
                    'maxiter': int(min(500, remaining_time * 5)),  # ~5 iterations per second
                    'xtol': 1e-8,
                    'ftol': 1e-8
                }
            )
            
            if res.success:
                pts = res.x.reshape(n, d)
                score = compute_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            continue

    # If we have time left, do a final polish on the best point
    if best_points is not None and time.time() < start_time + opt_budget - 30:
        try:
            x0 = best_points.flatten()
            res = minimize(
                smooth_objective,
                x0,
                method='Powell',
                options={
                    'maxiter': 200,
                    'xtol': 1e-10,
                    'ftol': 1e-10
                }
            )
            if res.success:
                pts = res.x.reshape(n, d)
                score = compute_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = initializations[0]

    return best_points

# EVOLVE-BLOCK-END
