# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize the configuration of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area.
    
    This implementation uses a local optimization approach (Nelder-Mead) starting
    from a regular 12-gon with a center point, which is known to outperform the
    regular 13-gon for the Heilbronn problem metric.
    """
    n = 13
    
    def calc_score(points):
        """Calculate the metric: min triangle area / hull area."""
        if not np.all(np.isfinite(points)):
            return -np.inf
        try:
            # Calculate areas of all 286 triangles
            # Vectorized calculation
            # points shape (13, 2)
            # Indices i, j, k
            idx = np.arange(n)
            # Create all combinations of 3 distinct indices
            # This is a bit slow to generate every time, so we can precompute
            # but for simplicity in the closure, we generate once outside?
            # No, let's just use a simple loop or vectorized trick.
            # Vectorized:
            # p1, p2, p3
            # Area = 0.5 * |det([p1-p3, p2-p3])|
            
            # Precompute indices
            i, j, k = np.indices((n, n, n))
            # Mask where i != j, i != k, j != k
            mask = (i != j) & (i != k) & (j != k)
            # We only need unique combinations to avoid duplicates, but min is same.
            # To save time, we can just compute all and take min.
            # 13^3 = 2197 calculations. Very fast.
            pi = points[i]
            pj = points[j]
            pk = points[k]
            
            # Area formula: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            # Or 0.5 * |(pj-pi) x (pk-pi)|
            v1 = pj - pi
            v2 = pk - pi
            # Cross product in 2D is scalar: v1_x * v2_y - v1_y * v2_x
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = np.abs(cross) * 0.5
            min_area = np.min(areas[mask])
            
            # Hull area
            hull = ConvexHull(points)
            hull_area = hull.volume
            
            if hull_area < 1e-9:
                return -np.inf
                
            return min_area / hull_area
        except Exception:
            return -np.inf

    def neg_score(x_flat):
        points = x_flat.reshape(n, 2)
        return -calc_score(points)

    # Precompute indices for speed if we were to optimize calc_score, 
    # but n=13 is small.
    
    # Initial Guess 1: Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    x_12 = np.cos(angles_12)
    y_12 = np.sin(angles_12)
    pts_12 = np.column_stack((x_12, y_12))
    center = np.array([0.0, 0.0])
    x0_1 = np.vstack([pts_12, center]).flatten()
    
    # Initial Guess 2: Regular 13-gon
    angles_13 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_13 = np.cos(angles_13)
    y_13 = np.sin(angles_13)
    x0_2 = np.column_stack((x_13, y_13)).flatten()
    
    # Initial Guess 3: Random (seeded)
    np.random.seed(42)
    x0_3 = np.random.rand(n, 2) * 2 - 1 # In [-1, 1]
    # Ensure not all collinear
    if np.linalg.matrix_rank(x0_3) < 2:
        x0_3[0, 1] += 0.1
        
    best_score = -np.inf
    best_points = None
    
    starts = [x0_1, x0_2, x0_3]
    
    for x0 in starts:
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to regular 13-gon
        return np.column_stack((np.cos(angles_13), np.sin(angles_13)))
        
    return best_points

# EVOLVE-BLOCK-END
