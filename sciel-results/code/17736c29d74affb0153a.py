import numpy as np
from scipy.spatial import ConvexHull

N = 13

def _score_configuration(points):
    """
    Calculate the score for a given set of points.
    Score = min_triangle_area / convex_hull_area
    """
    points = np.array(points, dtype=float)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for duplicate points
    # If duplicates exist, min area is 0
    # We can check this by computing all pair distances, but that's O(N^2).
    # For N=13, it's fine.
    # Alternatively, if ConvexHull works, we can just compute areas.
    # If points are duplicates, triangle area is 0.
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
        
    # Calculate min triangle area
    # Vectorized approach:
    # Create arrays for all combinations of 3 indices
    # i, j, k
    idx = np.arange(n)
    # We can use itertools.combinations to get indices, then vectorize the area calc
    # For N=13, 286 triangles. 
    # Let's stick to a loop for clarity and to avoid memory overhead of large arrays if N was bigger, 
    # but N is small.
    min_area = float('inf')
    
    # Pre-compute differences for speed?
    # area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    
    for i in range(n):
        p1 = points[i]
        for j in range(i+1, n):
            p2 = points[j]
            dx12 = p2[0] - p1[0]
            dy12 = p2[1] - p1[1]
            for k in range(j+1, n):
                p3 = points[k]
                dx13 = p3[0] - p1[0]
                dy13 = p3[1] - p1[1]
                area = 0.5 * abs(dx12 * dy13 - dx13 * dy12)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0 # Early exit if degenerate

    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def _generate_symmetric_points(k, m, r_inner, phi):
    """
    Generate 1 + k + m points.
    Center at (0,0).
    Layer 1: Regular k-gon at radius 1.0, angle offset 0.
    Layer 2: Regular m-gon at radius r_inner, angle offset phi.
    """
    center = np.array([[0.0, 0.0]])
    
    # Layer 1: k points, R=1.0
    angles1 = np.arange(k) * (2 * np.pi / k)
    pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    
    # Layer 2: m points, R=r_inner
    angles2 = np.arange(m) * (2 * np.pi / m) + phi
    pts2 = r_inner * np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    points = np.vstack([center, pts1, pts2])
    return points

def _generate_symmetric_points_reversed(k, m, r_inner, phi):
    """
    Generate 1 + k + m points.
    Center at (0,0).
    Layer 1: Regular k-gon at radius r_inner, angle offset phi.
    Layer 2: Regular m-gon at radius 1.0, angle offset 0.
    """
    center = np.array([[0.0, 0.0]])
    
    # Layer 2: m points, R=1.0 (This is the outer layer)
    angles2 = np.arange(m) * (2 * np.pi / m)
    pts2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    # Layer 1: k points, R=r_inner
    angles1 = np.arange(k) * (2 * np.pi / k) + phi
    pts1 = r_inner * np.column_stack([np.cos(angles1), np.sin(angles1)])
    
    points = np.vstack([center, pts1, pts2])
    return points

def construct():
    best_score = 0.0
    best_points = None
    
    # Pairs (k, m) such that 1 + k + m = 13
    # k, m >= 1
    # We consider k != m to avoid double counting, but the geometry of (k,m) vs (m,k) is different 
    # unless we swap the roles of inner/outer.
    # We will test both orientations for each unordered pair {k,m}.
    
    pairs = [
        (6, 6),
        (5, 7),
        (7, 5),
        (4, 8),
        (8, 4),
        (3, 9),
        (9, 3),
        (2, 10),
        (10, 2),
        (1, 11),
        (11, 1)
    ]
    
    # Filter out duplicates if we treat (k,m) and (m,k) as same structure?
    # No, (k,m) with k outer is different from (m,k) with m outer?
    # In _generate_symmetric_points, Layer 1 is always R=1.0.
    # So (k,m) means k is outer, m is inner.
    # (m,k) means m is outer, k is inner.
    # These are distinct configurations to search.
    
    for k, m in pairs:
        # Symmetry of phi:
        # The inner layer has m points. Rotating it by 2*pi/m yields the same set of points.
        # So phi in [0, 2*pi/m) is sufficient.
        phi_max = 2 * np.pi / m
        
        # Coarse Grid
        # r_inner in [0.1, 1.0]
        # phi in [0, phi_max]
        
        # Number of points: 1 + k + m = 13.
        # If k=1, "polygon" is a point.
        
        r_vals_coarse = np.linspace(0.1, 1.0, 20)
        phi_vals_coarse = np.linspace(0.0, phi_max, 20)
        
        for r in r_vals_coarse:
            for phi in phi_vals_coarse:
                points = _generate_symmetric_points(k, m, r, phi)
                score = _score_configuration(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
        # Fine Grid around best? Or global fine grid?
        # Global fine grid is safer for global optima.
        # 50 * 50 = 2500 evals per pair.
        # 11 pairs * 2500 = 27,500 evals.
        # Each eval is fast (N=13).
        # 27,500 * (286 triangle calcs) ~ 8M ops. Should be well under 120s.
        
        r_vals_fine = np.linspace(0.1, 1.0, 50)
        phi_vals_fine = np.linspace(0.0, phi_max, 50)
        
        for r in r_vals_fine:
            for phi in phi_vals_fine:
                points = _generate_symmetric_points(k, m, r, phi)
                score = _score_configuration(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()

    # Ensure we return a valid array
    if best_points is None:
        # Fallback: Return a regular 13-gon? Or the first point set?
        # A regular 13-gon is a valid convex set.
        angles = np.arange(13) * (2 * np.pi / 13)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
