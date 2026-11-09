# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _get_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _compute_score(pts, tri_idx):
    """Compute score given points and precomputed triangle indices."""
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _objective(pts, tri_idx):
    """Objective function for optimization (minimize negative score)."""
    score = _compute_score(pts, tri_idx)
    return -score

def _regular_polygon(n, radius=1.0, center=(0, 0), angle_offset=0.0):
    angles = angle_offset + np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _generate_initial_points(partition, tri_idx):
    """Generate initial points based on a partition of 13 into rings."""
    pts = []
    remaining = 13
    ring_counts = list(partition)
    
    for i, count in enumerate(ring_counts):
        if count == 0:
            continue
        if i == 0 and count == 1:
            # Center point
            pts.append([0.0, 0.0])
        else:
            # Distribute remaining rings evenly in radius
            radius = 1.0 / (len(ring_counts) - i) * (i + 1)
            # Add small jitter to break symmetry
            angle_offset = np.random.uniform(0, 2 * np.pi)
            poly = _regular_polygon(count, radius, (0, 0), angle_offset)
            pts.extend(poly)
            
    pts = np.array(pts)
    if len(pts) != 13:
        # Fallback to regular 13-gon
        pts = _regular_polygon(13)
    return pts

def _coordinate_descent(pts, tri_idx, max_iter=1000, step_size=0.01):
    """
    Coordinate descent with exact line search on each coordinate.
    """
    n = len(pts)
    best_score = _compute_score(pts, tri_idx)
    pts = pts.copy()
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for d in range(2):
                # Try to improve coordinate (i, d)
                best_val = pts[i, d]
                best_local_score = best_score
                
                # Binary search or golden section search for best step
                # We'll use a simple perturbation approach with shrinking step size
                for step in [step_size * 0.5**k for k in range(10)]:
                    for direction in [1, -1]:
                        new_pts = pts.copy()
                        new_pts[i, d] += direction * step
                        # Ensure points are within reasonable bounds to avoid numerical issues
                        if np.any(np.abs(new_pts) > 100):
                            continue
                        new_score = _compute_score(new_pts, tri_idx)
                        if new_score > best_local_score:
                            best_local_score = new_score
                            best_val = new_pts[i, d]
                            improved = True
                            
                if improved:
                    pts[i, d] = best_val
                    best_score = best_local_score
                    break # Restart loop for this point to account for changes
                    
        if not improved:
            break
            
    return pts, best_score

def _simulated_annealing(pts, tri_idx, time_limit=10.0):
    """
    Simulated annealing with coordinate moves.
    """
    start_time = time.time()
    n = len(pts)
    best_pts = pts.copy()
    best_score = _compute_score(best_pts, tri_idx)
    current_pts = pts.copy()
    current_score = best_score
    
    T = 0.1 # Initial temperature
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random point and coordinate
        i = np.random.randint(n)
        d = np.random.randint(2)
        
        # Random perturbation
        delta = np.random.normal(0, T * 0.1)
        new_pts = current_pts.copy()
        new_pts[i, d] += delta
        
        if np.any(np.abs(new_pts) > 100):
            continue
            
        new_score = _compute_score(new_pts, tri_idx)
        delta_score = new_score - current_score
        
        if delta_score > 0 or np.random.random() < np.exp(delta_score / T):
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
        T *= cooling
        
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    deadline = time.time() + 580  # Safety margin
    
    best_pts = None
    best_score = -1.0
    
    # 1. Try several symmetric initial configurations
    partitions = [
        (1, 12),
        (3, 10),
        (4, 9),
        (5, 8),
        (6, 7),
        (1, 3, 9),
        (1, 5, 7),
        (1, 6, 6),
        (2, 5, 6),
        (3, 4, 6),
        (2, 2, 9),
    ]
    
    # Regular 13-gon
    pts_reg = _regular_polygon(13)
    score_reg = _compute_score(pts_reg, tri_idx)
    if score_reg > best_score:
        best_score = score_reg
        best_pts = pts_reg.copy()
        
    for part in partitions:
        if time.time() > deadline - 100:
            break
        pts_init = _generate_initial_points(part, tri_idx)
        score_init = _compute_score(pts_init, tri_idx)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts_init.copy()
            
    # 2. Refine the best configuration using Coordinate Descent
    if best_pts is not None and time.time() < deadline - 50:
        pts_cd, score_cd = _coordinate_descent(best_pts, tri_idx, max_iter=200, step_size=0.05)
        if score_cd > best_score:
            best_score = score_cd
            best_pts = pts_cd.copy()
            
    # 3. Simulated Annealing from the best configuration
    if time.time() < deadline - 30:
        time_for_sa = min(30.0, deadline - time.time() - 10)
        pts_sa, score_sa = _simulated_annealing(best_pts, tri_idx, time_limit=time_for_sa)
        if score_sa > best_score:
            best_score = score_sa
            best_pts = pts_sa.copy()
            
    # 4. Final Coordinate Descent refinement
    if time.time() < deadline - 10:
        pts_final, score_final = _coordinate_descent(best_pts, tri_idx, max_iter=500, step_size=0.01)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final.copy()
            
    # Normalize to unit hull area (optional, score is scale invariant)
    hull_area = _hull_area(best_pts)
    if hull_area > 1e-9:
        best_pts = best_pts / np.sqrt(hull_area)
        
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        return _regular_polygon(13)
        
    return best_pts

# EVOLVE-BLOCK-END
