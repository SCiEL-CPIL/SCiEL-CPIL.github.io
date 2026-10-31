"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay
import itertools

H = np.sqrt(3) / 2
# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
VERTICES = np.array([A, B, C])

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _barycentric_coords(p):
    """Compute barycentric coordinates of point p relative to triangle ABC."""
    # Using the standard formula
    v0 = B - A
    v1 = C - A
    v2 = p - A
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return np.array([0.0, 0.0, 0.0])
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1.0 - u - v
    
    return np.array([u, v, w])

def _from_barycentric(u, v, w):
    """Convert barycentric coordinates to Cartesian."""
    return u * A + v * B + w * C

def _is_inside(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    bc = _barycentric_coords(p)
    return bc[0] >= -tol and bc[1] >= -tol and bc[2] >= -tol

def _is_valid(points):
    """Check if all points are inside the triangle."""
    return all(_is_inside(p) for p in points)

def _generate_structured_initial():
    """Generate initial points using a structured pattern based on Delaunay-like subdivision."""
    # Use a hierarchical subdivision approach
    # Level 0: 1 point (centroid)
    # Level 1: 3 points (midpoints of edges) + centroid = 4 points
    # Level 2: More subdivision
    
    # Strategy: Use a combination of vertices, edge midpoints, and interior points
    # For 11 points, use:
    # - 3 vertices (or near-vertices)
    # - 3 edge midpoints
    # - 5 interior points arranged in a pentagon-like pattern
    
    h = H
    
    # Near-vertices (slightly inside)
    eps = 0.02
    p1 = np.array([eps, eps * h])  # near A
    p2 = np.array([1 - eps, eps * h])  # near B
    p3 = np.array([0.5, h * (1 - eps)])  # near C
    
    # Edge midpoints
    m_ab = np.array([0.5, 0.0])
    m_bc = np.array([0.75, h * 0.5])
    m_ca = np.array([0.25, h * 0.5])
    
    # Interior points: use a regular pentagon scaled and centered at centroid
    centroid = (A + B + C) / 3.0
    r = 0.15  # radius of pentagon
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 2
    interior_points = []
    for a in angles:
        pt = centroid + r * np.array([np.cos(a), np.sin(a)])
        if _is_inside(pt):
            interior_points.append(pt)
        else:
            # Project back
            interior_points.append(centroid)
    
    points = [p1, p2, p3, m_ab, m_bc, m_ca] + interior_points
    
    # Ensure we have exactly 11 points
    while len(points) < 11:
        points.append(centroid)
    points = points[:11]
    
    return np.array(points)

def _generate_random_initial(seed=42):
    """Generate random initial points."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(11):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                p = _from_barycentric(1 - u - v, u, v)
                if _is_inside(p, tol=-1e-6):
                    points.append(p)
                    break
    return np.array(points)

def _delaunay_triangles(points):
    """Get the Delaunay triangulation triangles."""
    try:
        tri = Delaunay(points)
        triangles = points[tri.simplices]
        return triangles
    except:
        return None

def _objective(points_flat):
    """Objective function to minimize the negative of the minimum triangle area."""
    points = points_flat.reshape(-1, 2)
    if not _is_valid(points):
        return 1e10
    
    min_area = _min_triangle_area(points)
    return -min_area

def _multi_start_optimize():
    """Perform multi-start optimization with different initial configurations."""
    best_points = None
    best_area = -1
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Structured initial
    initial_configs.append(_generate_structured_initial())
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        initial_configs.append(_generate_random_initial(seed))
    
    # Also try some hand-crafted patterns
    # Pattern: 3 corners + 3 midpoints + 5 in a circle
    h = H
    centroid = (A + B + C) / 3.0
    for r in [0.1, 0.15, 0.2]:
        pts = []
        # Corners (slightly inside)
        eps = 0.03
        pts.append(np.array([eps, eps * h]))
        pts.append(np.array([1 - eps, eps * h]))
        pts.append(np.array([0.5, h * (1 - eps)]))
        # Midpoints
        pts.append(np.array([0.5, 0.0]))
        pts.append(np.array([0.75, h * 0.5]))
        pts.append(np.array([0.25, h * 0.5]))
        # Circle of 5
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        for a in angles:
            pt = centroid + r * np.array([np.cos(a), np.sin(a)])
            if _is_inside(pt):
                pts.append(pt)
            else:
                pts.append(centroid)
        if len(pts) >= 11:
            initial_configs.append(np.array(pts[:11]))
    
    # Optimize each initial config
    for init_pts in initial_configs:
        if not _is_valid(init_pts):
            continue
        
        init_area = _min_triangle_area(init_pts)
        
        # Local refinement using BFGS
        try:
            res = minimize(
                _objective,
                init_pts.flatten(),
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-12}
            )
            
            if res.success:
                refined_pts = res.x.reshape(-1, 2)
                if _is_valid(refined_pts):
                    refined_area = _min_triangle_area(refined_pts)
                    if refined_area > best_area:
                        best_area = refined_area
                        best_points = refined_pts
        except:
            pass
        
        # Also try differential evolution from this init
        try:
            bounds = [(0, 1), (0, H)] * 11
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                tol=1e-8,
                polish=False,
                init='sobol',
                popsize=10
            )
            
            de_pts = de_result.x.reshape(-1, 2)
            if _is_valid(de_pts):
                de_area = _min_triangle_area(de_pts)
                if de_area > best_area:
                    best_area = de_area
                    best_points = de_pts
        except:
            pass
    
    return best_points

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    best_points = _multi_start_optimize()
    
    if best_points is None or not _is_valid(best_points):
        # Fallback
        best_points = _generate_random_initial(7)
    
    # Final refinement
    try:
        res = minimize(
            _objective,
            best_points.flatten(),
            method='BFGS',
            options={'maxiter': 1000, 'gtol': 1e-14}
        )
        if res.success and _is_valid(res.x.reshape(-1, 2)):
            best_points = res.x.reshape(-1, 2)
    except:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
