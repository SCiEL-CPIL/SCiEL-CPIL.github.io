import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import differential_evolution, minimize, Bounds
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _voronoi_centroid(points, domain_radius=1.5):
    """
    Compute the centroid of each point's Voronoi cell clipped to a disk.
    Uses Delaunay triangulation to approximate Voronoi cells.
    """
    try:
        # Create Delaunay triangulation
        tri = Delaunay(points)
        
        # For each point, find its Voronoi cell by averaging the centroids of
        # adjacent Delaunay triangles, weighted by triangle area
        centroids = np.zeros_like(points)
        weights = np.zeros(N)
        
        for simplex in tri.simplices:
            # Get the three points of this triangle
            p0, p1, p2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Triangle centroid
            tri_centroid = (p0 + p1 + p2) / 3.0
            
            # Triangle area
            area = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))
            
            # Add to each vertex's weighted centroid
            for idx in simplex:
                centroids[idx] += tri_centroid * area
                weights[idx] += area
        
        # Normalize
        valid = weights > 1e-12
        for i in range(N):
            if valid[i]:
                centroids[i] /= weights[i]
            else:
                centroids[i] = points[i]
        
        return centroids
    except Exception:
        return points.copy()

def _voronoi_relax(points, iterations=50, domain_radius=1.5, cooling=0.95):
    """
    Iteratively move points toward their Voronoi cell centroids.
    This creates a more uniform distribution.
    """
    pts = points.copy()
    step_size = 0.5
    
    for it in range(iterations):
        centroids = _voronoi_centroid(pts, domain_radius)
        pts = pts + step_size * (centroids - pts)
        step_size *= cooling
        
        # Keep points within domain
        distances = np.linalg.norm(pts, axis=1)
        mask = distances > domain_radius
        if np.any(mask):
            pts[mask] = pts[mask] / distances[mask] * domain_radius
    
    return pts

def _affine_transform(points, shear_x, shear_y, rotation, scale):
    """
    Apply affine transformation: scale, shear, then rotation.
    """
    # Scale
    pts = points * scale
    
    # Shear
    shear_matrix = np.array([[1.0, shear_x], [shear_y, 1.0]])
    pts = pts @ shear_matrix.T
    
    # Rotation
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    pts = pts @ rot_matrix.T
    
    return pts

def _score_affine(params, base_points):
    """
    Score function for affine-transformed base points.
    params: [shear_x, shear_y, rotation, scale]
    """
    shear_x, shear_y, rotation, scale = params
    pts = _affine_transform(base_points, shear_x, shear_y, rotation, scale)
    return _calculate_score(pts)

def _optimize_affine(base_points, time_limit):
    """
    Optimize affine transformation parameters.
    """
    start_time = time.time()
    
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0, 2*np.pi), (0.5, 2.0)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search
    try:
        res = differential_evolution(
            lambda p: -_score_affine(p, base_points),
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Local polish
    if time.time() - start_time < time_limit * 0.7 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_affine(p, base_points),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _polish_points(points, time_limit, start_time):
    """
    Directly polish the point coordinates using Nelder-Mead on the exact objective.
    """
    deadline = start_time + time_limit
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Nelder-Mead on all 28 coordinates
    try:
        def neg_score(x):
            pts = x.reshape(-1, 2)
            return -_calculate_score(pts)
        
        res = minimize(
            neg_score,
            points.flatten(),
            method='Nelder-Mead',
            options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 2000, 'adaptive': True}
        )
        
        if time.time() < deadline:
            candidate_score = -res.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Second polish pass with tighter tolerance
    if time.time() < deadline - 2:
        try:
            def neg_score2(x):
                pts = x.reshape(-1, 2)
                return -_calculate_score(pts)
            
            res2 = minimize(
                neg_score2,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-15, 'maxiter': 1000, 'adaptive': True}
            )
            
            candidate_score = -res2.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = res2.reshape(-1, 2) if hasattr(res2, 'reshape') else res2.x.reshape(-1, 2)
        except Exception:
            pass
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy 1: Voronoi relaxation from regular polygon
    if time.time() < deadline - 80:
        # Start with regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        base_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Apply Voronoi relaxation
        relaxed_points = _voronoi_relax(base_points, iterations=30, domain_radius=1.5, cooling=0.9)
        
        # Optimize affine transformation
        t_limit = min(40.0, deadline - time.time() - 40)
        score, params = _optimize_affine(relaxed_points, t_limit)
        
        if score > best_overall_score and params is not None:
            best_overall_score = score
            best_overall_points = _affine_transform(relaxed_points, *params)
    
    # Strategy 2: Voronoi relaxation from perturbed random points
    if time.time() < deadline - 60 and best_overall_points is not None:
        np.random.seed(123)
        random_points = np.random.randn(N, 2) * 0.5
        # Ensure no points are too close
        for _ in range(100):
            if np.min(np.linalg.norm(random_points[:, None, :] - random_points[None, :, :], axis=2)) > 0.1:
                break
        
        relaxed_random = _voronoi_relax(random_points, iterations=40, domain_radius=1.5, cooling=0.92)
        
        t_limit = min(30.0, deadline - time.time() - 30)
        score, params = _optimize_affine(relaxed_random, t_limit)
        
        if score > best_overall_score and params is not None:
            best_overall_score = score
            best_overall_points = _affine_transform(relaxed_random, *params)
    
    # Strategy 3: Two-ring approach as alternative
    if time.time() < deadline - 40:
        n_out, n_in = 7, 7
        bounds = [(0.0, 0.999), (0.0, 2*np.pi)]
        
        try:
            res = differential_evolution(
                lambda p: -_calculate_score(np.vstack([
                    np.column_stack((np.cos(np.linspace(0, 2*np.pi, n_out, endpoint=False)),
                                     np.sin(np.linspace(0, 2*np.pi, n_out, endpoint=False)))),
                    np.column_stack((p[0]*np.cos(np.linspace(0, 2*np.pi, n_in, endpoint=False) + p[1]),
                                     p[0]*np.sin(np.linspace(0, 2*np.pi, n_in, endpoint=False) + p[1])))
                ])),
                bounds,
                seed=456,
                maxiter=25,
                popsize=12,
                tol=1e-5,
                polish=False
            )
            if res.success:
                r, theta = res.x
                angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                ring_points = np.vstack([
                    np.column_stack((np.cos(angles_out), np.sin(angles_out))),
                    np.column_stack((r*np.cos(angles_in), r*np.sin(angles_in)))
                ])
                ring_score = _calculate_score(ring_points)
                if ring_score > best_overall_score:
                    best_overall_score = ring_score
                    best_overall_points = ring_points
        except Exception:
            pass
    
    # Fallback: regular 14-gon
    if best_overall_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_overall_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_overall_score = _calculate_score(best_overall_points)
    
    # Final direct polish on all 28 coordinates
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 3
        polished_score, polished_points = _polish_points(
            best_overall_points, polish_time, start_time
        )
        if polished_score > best_overall_score:
            best_overall_score = polished_score
            best_overall_points = polished_points
    
    return best_overall_points
