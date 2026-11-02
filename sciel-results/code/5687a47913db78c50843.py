"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 13

# EVOLVE-BLOCK-START

def _score_points(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    pts: (N, 2) array of points.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas
    # We can vectorize this for speed
    # Number of triples: n*(n-1)*(n-2)/6
    # For n=13, that's 286 triangles.
    
    min_area = float('inf')
    
    # Vectorized computation of triangle areas
    # i < j < k
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p0 = pts[i]
    p1 = pts[j]
    p2 = pts[k]
    
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area


def _score_params(params):
    """
    Convert parameters to points and return the score.
    Parameters:
    - First 6: angles for a regular hexagon (fixed relative angles, but we can allow rotation)
      Actually, let's use a more flexible approach.
    
    Let's parameterize as:
    - 6 points on a regular hexagon with radius r1 and angle offset a1
    - 6 points on a regular hexagon with radius r2 and angle offset a2 (rotated by 30 degrees?)
    - 1 point at the center
    
    Wait, 6+6+1 = 13 points.
    
    Parameters: [r1, a1, r2, a2]
    - r1, r2: radii (positive)
    - a1, a2: angular offsets
    
    This enforces high symmetry and convexity (if r2 < r1, the outer hexagon is convex,
    and inner points are inside).
    """
    r1, a1, r2, a2 = params
    
    # Ensure positive radii
    r1 = max(0.1, r1)
    r2 = max(0.0, r2)
    
    # Outer hexagon
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a1
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    
    # Inner hexagon, rotated by 30 degrees (pi/6)
    angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a2 + np.pi/6
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    
    # Center point
    pts_center = np.array([[0.0, 0.0]])
    
    pts = np.vstack([pts1, pts2, pts_center])
    
    return _score_points(pts)


def construct():
    """
    Construct 13 points to maximize the minimum triangle area divided by convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit
    
    # Strategy 1: Symmetric hexagon-based configuration
    # Parameters: [r1, a1, r2, a2]
    # r1 in [0.5, 2.0], a1 in [0, 2*pi], r2 in [0, 1.0], a2 in [0, 2*pi]
    
    bounds = [
        (0.5, 2.0),  # r1
        (0, 2*np.pi),  # a1
        (0, 1.0),  # r2
        (0, 2*np.pi),  # a2
    ]
    
    # Use differential evolution for global search
    try:
        result = differential_evolution(
            _score_params,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42,
            tol=0.01,
            polish=True
        )
        
        best_params = result.x
        best_score = _score_params(best_params)
        
        # Check time
        if time.time() > deadline - 30.0:
            # Not enough time for refinement, return best found
            r1, a1, r2, a2 = best_params
            r1 = max(0.1, r1)
            r2 = max(0.0, r2)
            angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a1
            pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
            angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a2 + np.pi/6
            pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
            pts_center = np.array([[0.0, 0.0]])
            pts = np.vstack([pts1, pts2, pts_center])
            return pts
    except Exception:
        pass
    
    # Refine with local optimization
    if time.time() < deadline - 10.0:
        try:
            result2 = minimize(
                lambda x: -_score_params(x),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            best_params = result2.x
        except Exception:
            pass
    
    # Final computation
    r1, a1, r2, a2 = best_params
    r1 = max(0.1, r1)
    r2 = max(0.0, r2)
    
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a1
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    
    angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + a2 + np.pi/6
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    
    pts_center = np.array([[0.0, 0.0]])
    
    pts = np.vstack([pts1, pts2, pts_center])
    
    return pts

# EVOLVE-BLOCK-END
