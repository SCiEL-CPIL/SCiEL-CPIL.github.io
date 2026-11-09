# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Uses Simulated Annealing to find a high-quality configuration.
    """
    n = 13
    
    def triangle_areas(points):
        """
        Compute areas of all triangles formed by triples of points.
        Returns an array of shape (286,)
        """
        # points shape (n, 2)
        # We need all unique triples i < j < k
        # Vectorized approach for speed
        n_pts = len(points)
        # Create indices for all triples
        # This is a bit slow in pure python, but n=13 is small.
        # Let's use itertools for clarity, then vectorize the area calculation.
        import itertools
        triples = list(itertools.combinations(range(n_pts), 3))
        idx_i = np.array([t[0] for t in triples])
        idx_j = np.array([t[1] for t in triples])
        idx_k = np.array([t[2] for t in triples])
        
        p_i = points[idx_i]
        p_j = points[idx_j]
        p_k = points[idx_k]
        
        # Area = 0.5 * | (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)) |
        # Cross product of (p_j - p_i) and (p_k - p_i)
        v1 = p_j - p_i
        v2 = p_k - p_i
        # 2D cross product is scalar: v1_x * v2_y - v1_y * v2_x
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) / 2.0
        return areas

    def objective(points):
        """
        Objective function to minimize: negative of the ratio A_min / A_hull.
        """
        if not np.all(np.isfinite(points)):
            return 1e10
        
        # Check for duplicate points
        # If any two points are identical, min area is 0
        # We can check this efficiently
        # But let's rely on the area calculation. If area is 0, ratio is 0.
        
        areas = triangle_areas(points)
        a_min = np.min(areas)
        
        # Calculate Hull Area
        try:
            # Ensure points are not collinear
            # ConvexHull requires at least 3 non-collinear points
            hull = ConvexHull(points)
            a_hull = hull.volume
        except Exception:
            return 1e10
            
        if a_hull < 1e-10:
            return 1e10
            
        ratio = a_min / a_hull
        return -ratio

    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Simulated Annealing Parameters
    T_initial = 0.1
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    max_iter = 50000
    rng = np.random.default_rng(42)
    
    current_points = x0.copy()
    current_obj = objective(current_points)
    
    best_points = current_points.copy()
    best_obj = current_obj
    
    # Track time to stay well under 600s
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > 500: # Safety margin
            break
            
        T = T_initial * (alpha ** i)
        if T < T_min:
            break
            
        # Generate neighbor
        # Random step size
        step_size = 0.1 * (1 + T / T_initial)
        neighbor_points = current_points + rng.normal(0, step_size, (n, 2))
        
        new_obj = objective(neighbor_points)
        
        # Acceptance probability
        if new_obj < current_obj:
            # Better
            current_points = neighbor_points
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_points = current_points.copy()
        else:
            # Worse, accept with probability exp(-delta / T)
            delta = new_obj - current_obj
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_points = neighbor_points
                    current_obj = new_obj
    
    return best_points

# EVOLVE-BLOCK-END
