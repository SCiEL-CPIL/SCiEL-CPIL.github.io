import numpy as np
from scipy.optimize import minimize

N = 14

def points_from_radii(angles, radii):
    """Convert angles and radii to point coordinates."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def convex_hull_area(points):
    """Calculate area using shoelace formula (points in CCW order)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 distinct points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = abs(cross) * 0.5
                if area < min_area:
                    min_area = area
    return min_area

def objective(radii, angles):
    """Objective: minimize negative score."""
    if np.any(radii <= 0):
        return 1e6
    
    points = points_from_radii(angles, radii)
    
    area_hull = convex_hull_area(points)
    if area_hull < 1e-9:
        return 1e6
    
    area_min_tri = min_triangle_area(points)
    score = area_min_tri / area_hull
    return -score

def construct():
    # Non-uniform angles: cluster near diagonals (45, 135, 225, 315 degrees)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Define diagonal angles in radians
    diag_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    
    # Create non-uniform angles by warping: pull points toward diagonals
    angles = base_angles.copy()
    for diag in diag_angles:
        # Find nearest base angle to this diagonal
        idx = np.argmin(np.abs(angles - diag))
        # Shift nearby angles toward the diagonal
        for i in range(N):
            diff = (angles[i] - diag + np.pi) % (2 * np.pi) - np.pi
            if abs(diff) < np.pi / N * 3:
                angles[i] = angles[i] + 0.3 * (-diff)
    
    # Sort angles and normalize to [0, 2*pi)
    angles = np.sort(angles % (2 * np.pi))
    angles = np.sort(angles)
    if angles[0] < 0:
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        angles = np.sort(angles)
    
    best_score = -1e6
    best_points = None
    
    rng = np.random.default_rng(42)
    
    n_starts = 12
    
    for start_idx in range(n_starts):
        if start_idx == 0:
            radii0 = np.ones(N)
        else:
            base_radii = 1.0 + rng.normal(0, 0.1 * (1 + start_idx / n_starts), N)
            radii0 = np.maximum(base_radii, 0.3)
        
        result = minimize(
            objective,
            radii0,
            args=(angles,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        final_radii = result.x
        if np.any(final_radii <= 0):
            continue
        
        final_points = points_from_radii(angles, final_radii)
        
        area_hull = convex_hull_area(final_points)
        if area_hull > 1e-9:
            area_min_tri = min_triangle_area(final_points)
            score = area_min_tri / area_hull
            if score > best_score:
                best_score = score
                best_points = final_points
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        best_points = points_from_radii(angles, radii)
    
    return best_points
