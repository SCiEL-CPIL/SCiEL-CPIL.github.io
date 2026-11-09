import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate min triangle area
    min_area = float('inf')
    # There are C(13,3) = 286 triangles
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def _compute_min_triangle_and_indices(points):
    """
    Find the minimum area triangle and its indices.
    Returns (min_area, i, j, k)
    """
    min_area = float('inf')
    min_idx = (0, 1, 2)
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _fast_score(points):
    """
    Optimized scoring for the inner loop.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    min_area = float('inf')
    # Pre-computed indices for 13 points
    for i, j, k in _TRIS:
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if area < min_area:
            min_area = area
            if min_area < 1e-12:
                return 0.0 # Early exit if nearly collinear

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

# Pre-compute triangle indices for 13 points
_TRIS = [(i, j, k) for i in range(13) for j in range(i+1, 13) for k in range(j+1, 13)]

def _halton_sequence(n, base):
    """
    Generate the first n values of the Halton sequence in the given base.
    """
    seq = []
    f = 1.0
    i = 1
    while len(seq) < n:
        f /= base
        i += 1
        seq.append(f * (i % base))
        # This simple implementation is slow and incorrect for general base.
        # Let's use a better one.
    return seq

def _halton_sequence_correct(n, base):
    """
    Generate the first n values of the Halton sequence in the given base.
    """
    seq = []
    f = 1.0
    i = 1
    while len(seq) < n:
        f /= base
        i += 1
        # Standard Halton algorithm
        # Actually, let's just use a simple low-discrepancy sequence like Van der Corput
        # or just random if we want to keep it simple and robust.
        # For 13 points, random restarts with jitter are often sufficient.
        pass
    
    # Let's implement a proper Van der Corput / Halton
    def van_der_corput(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    seq = []
    for i in range(1, n+1):
        seq.append(van_der_corput(i, base))
    return seq

def _generate_initial_points(restart_idx):
    """
    Generate diverse initial points.
    """
    # Use Halton sequences for x and y
    # Base 2 for x, Base 3 for y
    x_vals = _halton_sequence_correct(13, 2)
    y_vals = _halton_sequence_correct(13, 3)
    
    # Scale to [-1, 1] roughly
    points = np.column_stack((x_vals * 2 - 1, y_vals * 2 - 1))
    
    # Add a small deterministic jitter based on restart_idx to break symmetry
    # Use a simple LCG or similar for reproducibility without numpy random state issues
    # Actually, using numpy random with a seed is fine.
    rng = np.random.default_rng(restart_idx)
    points += rng.normal(0, 0.1, (13, 2))
    
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Use a deterministic, geometry-aware optimization to find a configuration of 13 points
    that maximizes the ratio of the minimum triangle area to the convex hull area.
    
    Strategy:
    1. Multiple restarts with Halton-based initializations.
    2. In each restart, alternate between:
       a. Local Refinement: Identify the smallest triangle(s). Move points to maximize their area.
          - For the smallest triangle (i, j, k), move i, j, k to increase area.
          - Specifically, moving a point in the direction of the normal to the opposite side of the triangle
            (relative to the other two points) increases the area.
       b. Global Expansion: Push points outward from the centroid to increase the hull area.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 min budget, 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    num_restarts = 100 # We can do more restarts if each is fast
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            # Start with a regular polygon for the first one, as it's a good baseline
            theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
            current_points = np.column_stack((np.cos(theta), np.sin(theta)))
        else:
            current_points = _generate_initial_points(restart)
            
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Optimization Loop
        # We will run a fixed number of iterations or until time runs out.
        # Let's try 200 iterations per restart.
        max_iters = 200
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # 1. Find the minimum triangle
            min_area, (i_min, j_min, k_min) = _compute_min_triangle_and_indices(current_points)
            
            # 2. Local Refinement: Move points to increase min_area
            # We will move each of the 3 points in the direction that increases the area of the triangle.
            # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
            # To increase area, we can move p_i in the direction opposite to the normal of the edge (j,k) facing i?
            # Actually, the gradient of the area with respect to p_i is proportional to the vector (p_k - p_j) rotated by 90 degrees.
            # Let's just do a small step in the direction of the normal to the opposite side.
            
            # For point i, the opposite side is (j, k).
            # Vector v_jk = p_k - p_j
            # Normal n_jk = perp(v_jk)
            # We want to move p_i such that the distance from p_i to the line (j,k) increases.
            # The direction is the normal pointing away from the triangle.
            
            # Let's define a helper to get the "outward" normal for a point relative to a triangle.
            def get_expansion_dir(p, p_a, p_b):
                # Vector from a to b
                v = p_b - p_a
                # Normal (perp)
                # If v = (dx, dy), perp = (-dy, dx) or (dy, -dx)
                # We need the one pointing away from p.
                # Let's check the cross product (p - p_a) x v
                # If positive, p is on one side.
                # Let's just try both and pick the one that increases the area.
                # Actually, for a convex configuration, the "outward" normal is the one pointing away from the centroid.
                # But for a specific triangle, we want to push the vertex away from the opposite side.
                
                # Vector from midpoint of ab to p
                mid = (p_a + p_b) / 2
                vec = p - mid
                # Project vec onto normal of ab
                # Normal n = perp(v_ab)
                n1 = np.array([-v[1], v[0]])
                n2 = -n1
                
                # Which one points in the direction of vec?
                dot1 = np.dot(vec, n1)
                if dot1 > 0:
                    return n1 / (np.linalg.norm(n1) + 1e-10)
                else:
                    return n2 / (np.linalg.norm(n2) + 1e-10)

            # Apply small moves
            step_size = 0.05
            
            # Move i
            dir_i = get_expansion_dir(current_points[i_min], current_points[j_min], current_points[k_min])
            current_points[i_min] += step_size * dir_i
            
            # Move j
            dir_j = get_expansion_dir(current_points[j_min], current_points[i_min], current_points[k_min])
            current_points[j_min] += step_size * dir_j
            
            # Move k
            dir_k = get_expansion_dir(current_points[k_min], current_points[i_min], current_points[j_min])
            current_points[k_min] += step_size * dir_k
            
            # 3. Global Expansion: Push all points outward from centroid
            centroid = np.mean(current_points, axis=0)
            for idx in range(13):
                vec = current_points[idx] - centroid
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    current_points[idx] += 0.01 * (vec / norm)
            
            # 4. Evaluate
            new_score = _score(current_points)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # If it got worse, we might want to revert?
                # Or just continue?
                # Since this is a deterministic greedy step, if it fails, we are stuck.
                # Let's add a small random perturbation to escape local minima.
                # Or revert and try a different move?
                # For now, let's just continue. The next iteration will find the new min triangle.
                # To avoid getting stuck, we can occasionally add noise.
                if i % 10 == 0:
                    current_points += np.random.normal(0, 0.01, (13, 2))

    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        # Fallback: Regular polygon
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))

    return best_points

