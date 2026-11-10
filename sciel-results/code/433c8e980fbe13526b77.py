# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize
    the area of the smallest triangle formed by these points relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill
    
    def triangle_areas(points):
        """Compute all 286 triangle areas vectorized."""
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
                    areas.append(area)
        return np.array(areas)
    
    def soft_min_area(points, beta=0.01):
        """Smooth surrogate for minimum triangle area using log-sum-exp."""
        areas = triangle_areas(points)
        # Clip to avoid numerical issues
        areas = np.maximum(areas, 1e-10)
        # Soft minimum: -beta * log(sum(exp(-area/beta)))
        log_sum = np.log(np.sum(np.exp(-areas / beta)))
        return -beta * log_sum
    
    def hull_area(points):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 1.0  # fallback
    
    def objective(x, beta=0.01):
        """Objective to minimize: negative of (soft_min_area / hull_area)."""
        points = x.reshape((-1, 2))
        h_area = hull_area(points)
        if h_area < 1e-10:
            return 1e10
        sm_area = soft_min_area(points, beta)
        return -sm_area / h_area
    
    def get_score(points):
        """Compute the actual score: min_triangle_area / hull_area."""
        try:
            areas = triangle_areas(points)
            min_area = np.min(areas)
            h_area = hull_area(points)
            if h_area < 1e-10:
                return 0.0
            return min_area / h_area
        except Exception:
            return 0.0
    
    def polish(points):
        """Final polish: do a few iterations of exact optimization on the min triangle."""
        best = points.copy()
        best_score = get_score(best)
        
        # Find the minimum triangle
        n = len(best)
        min_idx = 0
        min_area = np.inf
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = best[i]
                    x2, y2 = best[j]
                    x3, y3 = best[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        
        i, j, k = min_idx
        # Try small perturbations of the three points in the min triangle
        for trial in range(50):
            if time.time() - start_time > deadline - 30:
                break
            # Random small perturbation
            delta = np.random.normal(0, 0.01, (3, 2))
            new_pts = best.copy()
            new_pts[i] += delta[0]
            new_pts[j] += delta[1]
            new_pts[k] += delta[2]
            s = get_score(new_pts)
            if s > best_score:
                best = new_pts
                best_score = s
        
        return best, best_score
    
    # Initial configurations
    def make_initial_config(rng):
        """Generate a random initial configuration."""
        config_type = rng.integers(0, 4)
        if config_type == 0:
            # Regular polygon with jitter
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.uniform(-0.3, 0.3, n)
            radii = 1.0 + rng.uniform(-0.2, 0.2, n)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif config_type == 1:
            # Jittered grid
            cols = 4
            rows = 4
            grid_x = np.linspace(-1, 1, cols)
            grid_y = np.linspace(-1, 1, rows)
            pts = np.array([[grid_x[i % cols], grid_y[i // cols]] for i in range(n)])
            pts += rng.normal(0, 0.1, pts.shape)
        elif config_type == 2:
            # Random in unit square
            pts = rng.uniform(-1, 1, (n, 2))
        else:
            # Two concentric rings
            inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + rng.uniform(-0.2, 0.2, 6)
            outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(-0.2, 0.2, 7)
            inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
            outer = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
            pts = np.vstack([inner, outer])
        return pts
    
    best_points = None
    best_score = -np.inf
    
    # Run multiple restarts
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 60:
            break
        
        # Generate initial configuration
        x0 = make_initial_config(rng)
        
        # Optimize with soft-min surrogate
        try:
            result = minimize(
                objective,
                x0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = result.x.reshape((-1, 2))
            
            # Polish
            x_polished, score = polish(x_opt)
            
            if score > best_score:
                best_score = score
                best_points = x_polished.copy()
        except Exception:
            continue
    
    # Final: try a few more polishes on the best found
    if best_points is not None:
        if time.time() - start_time < deadline - 30:
            best_points, best_score = polish(best_points)
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize: center and scale
    best_points = best_points - np.mean(best_points, axis=0)
    scale = np.max(np.abs(best_points))
    if scale > 0:
        best_points = best_points / scale
    
    return best_points

# EVOLVE-BLOCK-END
