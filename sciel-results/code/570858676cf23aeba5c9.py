import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for validity: finite, non-collinear
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    # Vectorized area calculation for all triangles
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the Heilbronn ratio.
    Strategy: Delaunay Triangulation Repair.
    1. Initialize with a regular 13-gon.
    2. Compute Delaunay triangulation.
    3. Iteratively identify the minimum area triangle in the Delaunay mesh.
    4. Apply a local "expansion" move: move the vertices of the min-area triangle
       slightly away from each other or towards the centroid of their neighbor
       triangles to increase the area, while keeping the hull roughly stable.
    5. Periodically re-normalize and apply a global "shape" optimization to fix
       drift and improve the hull area ratio.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Pre-compute triangle indices for speed
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i1, i2, i3 = idx[:, 0], idx[:, 1], idx[:, 2]
    
    def get_areas(pts):
        """Vectorized area calculation."""
        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]
        v1 = p2 - p1
        v2 = p3 - p1
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(det)

    def get_hull_area(pts):
        """Calculate convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

    def get_delaunay_triangles(pts):
        """
        Returns indices of triangles in the Delaunay triangulation.
        Only returns triangles where all 3 points are within the convex hull
        (which is always true for Delaunay of a point set, but we ensure
        we don't include degenerate or invalid ones).
        """
        try:
            tri = Delaunay(pts)
            # tri.simplices has shape (num_triangles, 3)
            # We filter out any simplices that might be invalid (though Delaunay usually handles this)
            return tri.simplices
        except Exception:
            return None

    def repair_step(pts, step_size=0.01):
        """
        Perform one Delaunay-based repair step.
        1. Find min area triangle in the full set (or Delaunay set).
        2. For that triangle (i, j, k), try to move points to increase area.
        3. A simple heuristic: move each vertex away from the centroid of the triangle.
        4. Also, consider the neighbors in the Delaunay mesh to avoid crossing edges.
        """
        areas = get_areas(pts)
        min_idx = np.argmin(areas)
        tri_idx = idx[min_idx]
        i, j, k = tri_idx
        
        # Get the centroid of the min-area triangle
        centroid = (pts[i] + pts[j] + pts[k]) / 3.0
        
        # Calculate vectors from centroid to vertices
        vi = pts[i] - centroid
        vj = pts[j] - centroid
        vk = pts[k] - centroid
        
        # Normalize directions
        ni = vi / (np.linalg.norm(vi) + 1e-12)
        nj = vj / (np.linalg.norm(vj) + 1e-12)
        nk = vk / (np.linalg.norm(vk) + 1e-12)
        
        # Proposed new positions: move slightly outward
        new_pts = pts.copy()
        new_pts[i] += ni * step_size
        new_pts[j] += nj * step_size
        new_pts[k] += nk * step_size
        
        # Check if this move improves the score or at least the min area
        # To be safe, we only accept if the global score doesn't drop significantly
        # However, calculating full score is expensive. 
        # Heuristic: Accept if the specific triangle area increased AND no other triangle became too small.
        
        old_min_area = areas[min_idx]
        new_areas = get_areas(new_pts)
        new_min_area = np.min(new_areas)
        
        # Accept if the global minimum area increased, or if the specific triangle improved significantly
        # and the global minimum didn't drop.
        if new_min_area > old_min_area:
            return new_pts, True
        
        # Alternative move: try moving towards the centroid of the *neighbors*
        # This is complex. Let's try a simpler "repulsion" from all other points.
        # If the above failed, try a small random perturbation on these 3 points
        # weighted by their current contribution to the min area.
        
        rng = np.random.default_rng()
        for _ in range(10): # Try a few random perturbations
            perturb = rng.normal(0, step_size, (3, 2))
            trial_pts = pts.copy()
            trial_pts[i] += perturb[0]
            trial_pts[j] += perturb[1]
            trial_pts[k] += perturb[2]
            
            trial_areas = get_areas(trial_pts)
            if np.min(trial_areas) > old_min_area:
                return trial_pts, True
                
        return pts, False

    def global_optimize(pts, max_iter=20):
        """
        A simple coordinate descent / gradient-free polish to fix the hull
        and maximize the ratio.
        """
        current_score = compute_score(pts)
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(n):
                for j in range(2):
                    # Try small steps in each dimension
                    for delta in [0.01, -0.01, 0.001, -0.001]:
                        trial = pts.copy()
                        trial[i, j] += delta
                        if not np.all(np.isfinite(trial)):
                            continue
                        score = compute_score(trial)
                        if score > current_score:
                            pts = trial
                            current_score = score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return pts

    # Initialize
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area hull?
    # The score is scale invariant, but numerical stability might benefit from fixed scale.
    
    best_points = points.copy()
    best_score = compute_score(best_points)
    
    # Main loop
    iterations = 0
    while time.time() < deadline - 20:
        iterations += 1
        
        # 1. Delaunay Repair Step
        # We use a decaying step size
        step_size = 0.05 / (1 + iterations * 0.1)
        new_points, accepted = repair_step(points, step_size)
        
        if accepted:
            points = new_points
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # If repair failed, try a global optimization step
            # This is expensive, so we do it less frequently
            if iterations % 10 == 0:
                points = global_optimize(points, max_iter=5)
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        # 2. Periodic Re-centering and Normalization
        # Keep points centered at origin
        points -= np.mean(points, axis=0)
        
        # 3. Occasionally, try a "jump" to a different topology
        # E.g., move an interior point to the hull or vice versa
        # For 13 points, a regular polygon is often optimal or close.
        # If we stagnate, try a random restart from a perturbed regular polygon.
        if iterations % 50 == 0:
            # Check stagnation
            if best_score < 55.0: # Threshold for restart
                angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.random.uniform(-0.1, 0.1)
                points = np.column_stack([np.cos(angles), np.sin(angles)])
                points -= np.mean(points, axis=0)

    return best_points

# EVOLVE-BLOCK-END
