"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.spatial import ConvexHull
import sys

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_rings(n, inner_frac=0.5):
    """Generate points on two concentric rings."""
    # Try to split n into two groups
    n_outer = n // 2
    n_inner = n - n_outer
    
    outer_angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner  # offset
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = inner_frac * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])

def _grid_like(n):
    """Generate a grid-like arrangement."""
    # For 14 points, try a 4x4 grid with one removed, or similar
    # Let's try a more structured approach
    # 14 points: maybe 2 rows of 7? Or a hexagonal packing?
    
    # Try hexagonal packing
    points = []
    # Center
    points.append([0, 0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Second ring (7 points to make 14 total)
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi/7
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    
    return np.array(points)

def _optimize_local(points, max_iters=5000, step_size=0.01, deadline=None):
    """Simple local optimization using gradient-free method."""
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random perturbation
        idx = np.random.randint(len(points))
        delta = np.random.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Decay step size occasionally
            if it % 100 == 0:
                step_size *= 0.9
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with several different configurations
    candidates = []
    
    # Regular polygon
    candidates.append(_regular_polygon(N))
    
    # Two rings
    candidates.append(_two_rings(N, 0.5))
    candidates.append(_two_rings(N, 0.6))
    candidates.append(_two_rings(N, 0.7))
    
    # Grid-like
    candidates.append(_grid_like(N))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        candidates.append(rng.rand(N, 2))
    
    # Evaluate initial candidates
    best_points = None
    best_score = -1
    
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Local optimization from the best candidate
    if best_points is not None and time.time() < deadline:
        best_points, best_score = _optimize_local(
            best_points, 
            max_iters=10000, 
            step_size=0.05, 
            deadline=deadline
        )
    
    # Try optimization from other candidates if time permits
    for pts in candidates:
        if time.time() > deadline - 10:
            break
        if np.allclose(pts, best_points):
            continue
        opt_pts, opt_score = _optimize_local(
            pts, 
            max_iters=5000, 
            step_size=0.05, 
            deadline=deadline
        )
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Final check
    if time.time() > deadline - 5:
        return best_points if best_points is not None else _regular_polygon(N)
    
    # One more round of fine-tuning
    if best_points is not None:
        best_points, best_score = _optimize_local(
            best_points, 
            max_iters=5000, 
            step_size=0.01, 
            deadline=deadline
        )
    
    return best_points if best_points is not None else _regular_polygon(N)
# EVOLVE-BLOCK-END
