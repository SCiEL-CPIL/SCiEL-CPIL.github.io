"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

def _triangle_areas_fast(points):
    """Compute min area of all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = _triangle_areas_fast(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area

def _generate_points(params):
    """
    Generate 13 points from angular Fourier perturbation.
    params: [A1, phi1, A2, phi2]
    theta_k = 2*pi*k/13 + A1*cos(k + phi1) + A2*cos(2*k + phi2)
    Radius is fixed to 1.
    """
    n = N
    A1, phi1, A2, phi2 = params
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    pert1 = A1 * np.cos(1 * base_angles + phi1)
    pert2 = A2 * np.cos(2 * base_angles + phi2)
    
    angles = base_angles + pert1 + pert2
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points

def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    
    # Check convexity
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            return -0.01
    except Exception:
        return -0.01
        
    return -_score(points)

def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses Differential Evolution and Nelder-Mead on a 4-parameter angular Fourier model.
    """
    # Bounds: [A1, phi1, A2, phi2]
    bounds = [
        (-0.2, 0.2),      # A1
        (0, 2 * np.pi),   # phi1
        (-0.2, 0.2),      # A2
        (0, 2 * np.pi),   # phi2
    ]
    
    best_score = 0.0
    best_points = None
    
    # Fallback: Regular Polygon
    angles_reg = np.arange(N) * 2 * np.pi / N
    points_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    best_score = _score(points_reg)
    best_points = points_reg
    
    try:
        # Global Optimization
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-5,
            polish=False
        )
        
        best_params = result.x
        best_points_de = _generate_points(best_params)
        
        # Verify convexity
        try:
            hull = ConvexHull(best_points_de)
            if len(hull.vertices) >= N:
                score_de = _score(best_points_de)
                if score_de > best_score:
                    best_score = score_de
                    best_points = best_points_de
        except Exception:
            pass
        
        # Local Refinement using Nelder-Mead from the best DE result
        if best_score > 0:
            result_local = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            best_params_local = result_local.x
            best_points_local = _generate_points(best_params_local)
            
            try:
                hull = ConvexHull(best_points_local)
                if len(hull.vertices) >= N:
                    score_local = _score(best_points_local)
                    if score_local > best_score:
                        best_score = score_local
                        best_points = best_points_local
            except Exception:
                pass
                
    except Exception:
        pass
        
    return best_points

# EVOLVE-BLOCK-END
