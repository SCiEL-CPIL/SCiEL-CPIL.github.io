# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the moving sofa path using differential evolution with a fast
    geometric collision oracle based on the Separating Axis Theorem (SAT).
    """
    
    # Sofa grid points (as defined in the task)
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Precompute the convex hull of the sofa for fast collision testing
    from scipy.spatial import ConvexHull
    hull = ConvexHull(sofa_points)
    hull_vertices = sofa_points[hull.vertices]
    n_hull = len(hull_vertices)
    
    # Base corridor vertices (L-shaped, width 1, inner corner at origin)
    # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    # The base corridor is a polygon. Let's define its vertices in CCW order.
    # The L-shape has vertices:
    # (-3, 0), (-3, 1), (0, 1), (0, -3), (1, -3), (1, 1), (0, 1) -- wait, let me trace carefully.
    # Rectangle 1: 0<=u<=1, -3<=v<=1 -> corners (0,-3),(1,-3),(1,1),(0,1)
    # Rectangle 2: -3<=u<=1, 0<=v<=1 -> corners (-3,0),(-3,1),(1,1),(1,0)
    # Union: The boundary goes:
    # Start at (-3, 0) -> (-3, 1) -> (1, 1) -> (1, -3) -> (0, -3) -> (0, 0) -> (-3, 0)
    # Wait, that's not right. Let me think again.
    # The union is an L-shape. The outer boundary in CCW:
    # (-3, 0) -> (-3, 1) -> (1, 1) -> (1, -3) -> (0, -3) -> (0, 0) -> (-3, 0)
    # But wait, is (0,0) to (-3,0) part of the boundary? The second rect is -3<=u<=1, 0<=v<=1.
    # So the bottom edge of the second rect is from (-3,0) to (1,0).
    # The first rect is 0<=u<=1, -3<=v<=1. Its left edge is from (0,-3) to (0,1).
    # The union's boundary:
    # Start at (-3, 0). Go up to (-3, 1). Go right to (1, 1). Go down to (1, -3). 
    # Go left to (0, -3). Go up to (0, 0). Go left to (-3, 0).
    # Yes, that's correct. 6 vertices.
    base_corridor = np.array([
        [-3, 0],
        [-3, 1],
        [1, 1],
        [1, -3],
        [0, -3],
        [0, 0]
    ])
    
    def is_point_in_base_corridor(point):
        """Check if a point (in base corridor coordinates) is inside the L-shaped corridor."""
        u, v = point
        # Inside rect 1: 0 <= u <= 1, -3 <= v <= 1
        # Inside rect 2: -3 <= u <= 1, 0 <= v <= 1
        in_rect1 = (0 <= u <= 1) and (-3 <= v <= 1)
        in_rect2 = (-3 <= u <= 1) and (0 <= v <= 1)
        return in_rect1 or in_rect2
    
    def sofa_fits_in_placement(x, y, a_deg):
        """
        Check if all sofa points fit inside the corridor placement (x, y, a_deg).
        Uses the convex hull for efficiency, but we need to check all points since
        the sofa is not convex. Actually, the sofa IS the convex hull of its points
        since it's a solid rectangle. Let me verify: the sofa is a 4.5 x 3.0 rectangle,
        which is convex. So checking the hull vertices is sufficient for collision.
        """
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Rotation matrix R(-a): 
        # [cos(a), sin(a); -sin(a), cos(a)]
        # Wait, R(-a) applied to (w - (x,y)):
        # If R(theta) is standard rotation by theta, then R(-a) = R(-a)
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        
        # Transform sofa points to corridor frame
        # w' = R(-a) * (w - (x, y))
        # w' = [[cos(a), sin(a)], [-sin(a), cos(a)]] * (w - (x, y))
        
        # For efficiency, transform only hull vertices first
        # Since the sofa is convex, if all hull vertices are inside, the whole sofa is inside
        
        # Transform hull vertices
        dx = hull_vertices[:, 0] - x
        dy = hull_vertices[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Check if all hull vertices are inside the base corridor
        # Use vectorized check
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        inside = in_rect1 | in_rect2
        
        return np.all(inside)
    
    def evaluate_path(path_params):
        """
        Evaluate a path given by parameters.
        path_params: array of shape (n, 3) with (x, y, a) for each intermediate point.
        """
        n_points = len(path_params)
        if n_points == 0:
            return 0.0
        
        # Build full path with start and end
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        
        # Intermediate points
        intermediate = path_params.copy()
        
        # Densify: insert linear interpolations between consecutive points
        # m = max(1, ceil(|a2 - a1|)) equal steps
        full_path = [start]
        
        all_points = [start] + list(intermediate) + [end]
        
        for i in range(len(all_points) - 1):
            p1 = all_points[i]
            p2 = all_points[i + 1]
            
            if i == 0:
                # Start is already added
                pass
            
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            
            if m > 1:
                for k in range(1, m + 1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    full_path.append(interp)
            else:
                full_path.append(p2)
        
        # Check all placements
        for x, y, a in full_path:
            if not sofa_fits_in_placement(x, y, a):
                return 0.0
        
        # If all placements fit, compute the area
        # The sofa is the intersection of all corridor placements
        # For each sofa point, check if it's inside every placement
        # This is expensive, so we do it only when the path is valid
        
        n_sofa = len(sofa_points)
        inside_all = np.ones(n_sofa, dtype=bool)
        
        for x, y, a in full_path:
            a_rad = np.radians(a)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = sofa_points[:, 0] - x
            dy = sofa_points[:, 1] - y
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2
            
            inside_all &= inside
        
        n_inside = np.sum(inside_all)
        return n_inside * 0.0001
    
    # Optimize the path using differential evolution
    # We'll optimize 5 intermediate points (x, y, a) each
    n_intermediate = 5
    n_params = n_intermediate * 3
    
    bounds = [
        (-1.0, 0.0),   # x1
        (-0.5, 0.5),   # y1
        (10.0, 80.0),  # a1
        (-1.0, 0.0),   # x2
        (-0.5, 0.5),   # y2
        (10.0, 80.0),  # a2
        (-1.0, 0.0),   # x3
        (-0.5, 0.5),   # y3
        (10.0, 80.0),  # a3
        (-1.0, 0.0),   # x4
        (-0.5, 0.5),   # y4
        (10.0, 80.0),  # a4
        (-1.0, 0.0),   # x5
        (-0.5, 0.5),   # y5
        (10.0, 80.0),  # a5
    ]
    
    def objective(params):
        path_params = params.reshape(n_intermediate, 3)
        return -evaluate_path(path_params)  # Minimize negative area
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        polish=False,
        update_interval=10,
    )
    
    # Extract the best path
    best_params = result.x.reshape(n_intermediate, 3)
    
    # Ensure angles are in increasing order
    best_params = best_params[np.argsort(best_params[:, 2])]
    
    return best_params


# EVOLVE-BLOCK-END
