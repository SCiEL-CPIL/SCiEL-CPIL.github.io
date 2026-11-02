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
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
        
    # Calculate min triangle area
    # Vectorized approach for all triangles
    # Number of triangles is C(13, 3) = 286
    # We can compute all areas at once using broadcasting or loops.
    # Given N=13, a simple loop is fast enough, but let's try to optimize slightly.
    
    min_area = float('inf')
    
    # Pre-compute differences? No, just direct calculation.
    for i in range(n):
        p1 = points[i]
        for j in range(i+1, n):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j+1, n):
                p3 = points[k]
                v2 = p3 - p1
                # Cross product magnitude in 2D
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0

    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def _generate_points(r1, phi1, r2, phi2):
    """
    Generate 13 points:
    1 center
    6 points at radius r1, offset phi1
    6 points at radius r2, offset phi2
    """
    angles_base = np.arange(6) * (np.pi / 3)
    
    # Center
    center = np.array([[0.0, 0.0]])
    
    # Ring 1
    angles1 = angles_base + phi1
    pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    
    # Ring 2
    angles2 = angles_base + phi2
    pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    points = np.vstack([center, pts1, pts2])
    return points

def construct():
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Search space:
    # r1, r2 in (0, 1)
    # phi1, phi2 in [0, pi/3)
    # We assume r1 < r2 is not required, but due to symmetry of the problem 
    # (swapping rings doesn't change the set of points if we also swap params),
    # we can restrict search to r1 <= r2 to halve the space, or just search full space.
    # However, phi1 and phi2 are independent.
    
    # Coarse Grid
    # 10x10x10x10 = 10,000 evaluations.
    # Each evaluation involves ConvexHull (fast for 13 pts) and 286 triangle area calcs.
    # Should be well within 120s.
    
    r_vals = np.linspace(0.1, 0.9, 10)
    phi_vals = np.linspace(0.05, np.pi/3 - 0.05, 10)
    
    # We'll iterate over all combinations
    # To ensure we don't miss the boundary where r1=r2 or phi1=phi2, 
    # but those are measure zero and likely degenerate or suboptimal.
    
    for r1 in r_vals:
        for r2 in r_vals:
            for phi1 in phi_vals:
                for phi2 in phi_vals:
                    points = _generate_points(r1, phi1, r2, phi2)
                    score = _score_configuration(points)
                    if score > best_score:
                        best_score = score
                        best_params = (r1, phi1, r2, phi2)
                        best_points = points.copy()
                        
    # Fine Grid around best coarse solution
    if best_params:
        r1_c, phi1_c, r2_c, phi2_c = best_params
        
        # Define fine ranges around the coarse best
        # Step size 1/10 of the coarse step
        coarse_step_r = r_vals[1] - r_vals[0]
        coarse_step_phi = phi_vals[1] - phi_vals[0]
        
        fine_step_r = coarse_step_r / 10
        fine_step_phi = coarse_step_phi / 10
        
        # 11 points in each dimension -> 11^4 = 14641 evaluations.
        r1_fine = np.linspace(max(0.01, r1_c - 2*fine_step_r), min(0.99, r1_c + 2*fine_step_r), 11)
        r2_fine = np.linspace(max(0.01, r2_c - 2*fine_step_r), min(0.99, r2_c + 2*fine_step_r), 11)
        phi1_fine = np.linspace(max(0.0, phi1_c - 2*fine_step_phi), min(np.pi/3, phi1_c + 2*fine_step_phi), 11)
        phi2_fine = np.linspace(max(0.0, phi2_c - 2*fine_step_phi), min(np.pi/3, phi2_c + 2*fine_step_phi), 11)
        
        for r1 in r1_fine:
            for r2 in r2_fine:
                for phi1 in phi1_fine:
                    for phi2 in phi2_fine:
                        points = _generate_points(r1, phi1, r2, phi2)
                        score = _score_configuration(points)
                        if score > best_score:
                            best_score = score
                            best_params = (r1, phi1, r2, phi2)
                            best_points = points.copy()
                            
    # Ensure we return a valid array
    if best_points is None:
        # Fallback: Regular 13-gon
        angles = np.arange(13) * (2 * np.pi / 13)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
