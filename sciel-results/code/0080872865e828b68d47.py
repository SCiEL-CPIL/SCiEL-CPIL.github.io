"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math


def _covering_radius(pts):
    """Compute the covering radius (in radians) for points on the unit sphere.
    Uses the convex hull: the covering radius is the max circumradius of facets
    that have an empty cap (i.e., the origin is on the correct side).
    """
    pts = np.asarray(pts)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms

    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute max angular distance to nearest point on a grid
        return _covering_radius_grid(pts)

    # For each facet, compute the circumradius of the spherical cap
    # The covering radius is the maximum over all facets of the angular distance
    # from the facet's center to the farthest vertex of that facet, but only for
    # facets where the cap is empty (no other points inside).
    
    # Actually, the covering radius equals the maximum over all points y on the sphere
    # of the angular distance to the nearest point. This is equivalent to the maximum
    # circumradius of the Delaunay triangulation on the sphere.
    # 
    # For the convex hull, each facet corresponds to a cell in the Voronoi diagram.
    # The circumcenter of a facet (in 3D) projected onto the sphere gives the center
    # of the corresponding Voronoi cell. The covering radius for that cell is the
    # angular distance from this center to the farthest vertex of the facet.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        # Check if the origin is on the inside (empty cap condition)
        # The facet normal should point away from the origin for a valid covering
        if len(facet_pts) >= 3:
            # Compute the circumcenter of the facet in 3D
            # For a triangle, the circumcenter is the intersection of perpendicular bisectors
            a, b, c = facet_pts[:3] if len(facet_pts) == 3 else (facet_pts[0], facet_pts[1], facet_pts[2])
            
            # Circumcenter of triangle abc
            # Using the formula: circumcenter = (|a|^2(b×c) + |b|^2(c×a) + |c|^2(a×b)) / (2(a·(b×c)))
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            ab = np.cross(a, b)
            
            denom = 2.0 * np.dot(a, bc)
            if abs(denom) < 1e-12:
                continue
            
            center = (np.dot(a, a) * bc + np.dot(b, b) * ca + np.dot(c, c) * ab) / denom
            
            # Project center onto unit sphere
            center_norm = np.linalg.norm(center)
            if center_norm < 1e-12:
                continue
            center_unit = center / center_norm
            
            # Check empty cap: all other points should be on the same side of the facet
            # as the origin (or equivalently, the facet normal points away from origin)
            # The facet normal (pointing outward from convex hull) should satisfy
            # n · origin < 0, i.e., n · 0 = 0... 
            # Better: check that center_unit is "outside" all other points, i.e.,
            # for all other points p, dot(p, center_unit) <= dot(facet_vertex, center_unit)
            # Actually, the empty cap means no point lies in the cap centered at center_unit
            # with angular radius = covering radius of this cell.
            
            # For the circumradius of this cell: it's the max angular distance from center_unit
            # to any vertex of the facet
            for v in facet_pts:
                cos_angle = np.clip(np.dot(center_unit, v), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle > max_radius:
                    max_radius = angle
    return max_radius


def _covering_radius_grid(pts, n_theta=200, n_phi=400):
    """Fallback: compute covering radius using a grid of points on the sphere."""
    pts = np.asarray(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to cartesian
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find the nearest configuration point
    max_dist = 0.0
    chunk_size = 10000
    for i in range(0, len(grid), chunk_size):
        chunk = grid[i:i+chunk_size]
        # dot products
        dots = chunk @ pts.T  # (chunk_size, N)
        # angular distance = arccos(dot), but we can just track max dot (min distance)
        min_dots = np.max(dots, axis=1)  # max dot = min angle
        angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
        if np.max(angles) > max_dist:
            max_dist = np.max(angles)
    return max_dist


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin before 120s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_radius = _covering_radius_grid(best_pts)
    
    # Gradient-based optimization using smooth surrogate
    # Objective: minimize the soft-max of angular distances to nearest point
    # For a point y on sphere, d(y) = min_i arccos(y·x_i)
    # We want to minimize max_y d(y)
    # 
    # Alternative: minimize the sum of exp(k * d(y_i)) for a set of test points,
    # or use the fact that the covering radius equals the max circumradius of hull facets.
    #
    # Let's use a different approach: minimize a smooth approximation of the covering radius
    # by considering the potential function.
    
    # Use L-BFGS-B or similar to minimize a smooth surrogate
    # The covering radius can be approximated by considering points at the vertices of the
    # Voronoi cells (circumcenters of hull facets).
    
    def objective_and_grad(params):
        """params: (N, 3) flattened. Returns (value, gradient) in radians."""
        pts = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts_n = pts / norms
        
        # Compute covering radius using grid (smooth enough for optimization)
        # Use a moderate grid for speed
        n_theta = 120
        n_phi = 240
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(TH) * np.cos(PH)
        y = np.sin(TH) * np.sin(PH)
        z = np.cos(TH)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # For each grid point, compute angular distance to nearest point
        dots = grid @ pts_n.T  # (M, N)
        # Soft-min: -logsumexp(-k * dots) / k approximates min(dots) = cos(max_angle)
        # Actually we want max angle = arccos(min_i cos(angle_i)) = arccos(max_i... no
        # angle_i = arccos(grid @ pts_i), we want max_i angle_i = arccos(min_i cos(angle_i))
        # min_i cos(angle_i) = min_i dots[i,j]
        # So covering radius ≈ arccos(min_j dots[:,j]) for each grid point, then max over grid
        # 
        # For smoothness, use soft-min of dots: soft_min_k(dots) = -(1/k) log(sum(exp(-k*dots)))
        # This approximates min(dots) as k -> inf
        
        k = 50.0
        neg_k_dots = -k * dots
        # log-sum-exp trick
        max_neg = np.max(neg_k_dots, axis=1, keepdims=True)
        log_sum = max_neg.ravel() + np.log(np.sum(np.exp(neg_k_dots - max_neg), axis=1))
        soft_min_dots = -log_sum / k
        
        # Covering radius approx = arccos(soft_min_dots)
        # But we want to minimize the max over grid points
        # Use soft-max over grid points of arccos(soft_min_dots)
        
        angles = np.arccos(np.clip(soft_min_dots, -1.0, 1.0))
        
        # Soft-max of angles
        k2 = 20.0
        max_ang = np.max(angles)
        log_sum_exp = max_ang + np.log(np.sum(np.exp(k2 * (angles - max_ang))))
        soft_max_angle = (log_sum_exp) / k2
        
        # Gradient with respect to pts_n
        # This is complex; let's use finite differences or a simpler approach
        # Actually, let's just use the value and compute gradient numerically
        
        return soft_max_angle
    
    # Use a simpler optimization: coordinate descent or direct L-BFGS with numerical gradient
    # Let's use scipy minimize with BFGS and numerical gradient
    
    def smooth_objective(params):
        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts_n = pts / norms
        
        # Use a grid to estimate covering radius
        n_theta = 100
        n_phi = 200
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(TH) * np.cos(PH)
        y = np.sin(TH) * np.sin(PH)
        z = np.cos(TH)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        dots = grid @ pts_n.T
        k = 40.0
        neg_k_dots = -k * dots
        max_neg = np.max(neg_k_dots, axis=1, keepdims=True)
        log_sum = max_neg.ravel() + np.log(np.sum(np.exp(neg_k_dots - max_neg), axis=1))
        soft_min_dots = -log_sum / k
        angles = np.arccos(np.clip(soft_min_dots, -1.0, 1.0))
        
        k2 = 15.0
        max_ang = np.max(angles)
        log_sum_exp = max_ang + np.log(np.sum(np.exp(k2 * (angles - max_ang))))
        return log_sum_exp / k2
    
    # Optimize
    x0 = best_pts.ravel()
    
    result = minimize(
        smooth_objective, 
        x0, 
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
    )
    
    if result.success or result.fun < best_radius:
        opt_pts = result.x.reshape(N, 3)
        norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
        opt_pts = opt_pts / norms
        opt_radius = _covering_radius_grid(opt_pts)
        if opt_radius < best_radius:
            best_pts = opt_pts
            best_radius = opt_radius
    
    # Check time
    if time.time() > deadline - 20:
        return best_pts
    
    # Try multiple random restarts with local optimization
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        
        # Random start
        random_pts = rng.standard_normal((N, 3))
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        # Local optimization
        result = minimize(
            smooth_objective,
            random_pts.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if result.fun < best_radius:
            opt_pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            opt_pts = opt_pts / norms
            opt_radius = _covering_radius_grid(opt_pts)
            if opt_radius < best_radius:
                best_pts = opt_pts
                best_radius = opt_radius
    
    # Final polish with higher precision grid
    if time.time() < deadline - 10:
        # Do a final optimization with finer grid
        def fine_objective(params):
            pts = params.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts_n = pts / norms
            
            n_theta = 150
            n_phi = 300
            theta = np.linspace(0, np.pi, n_theta)
            phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
            TH, PH = np.meshgrid(theta, phi, indexing='ij')
            x = np.sin(TH) * np.cos(PH)
            y = np.sin(TH) * np.sin(PH)
            z = np.cos(TH)
            grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            
            dots = grid @ pts_n.T
            k = 50.0
            neg_k_dots = -k * dots
            max_neg = np.max(neg_k_dots, axis=1, keepdims=True)
            log_sum = max_neg.ravel() + np.log(np.sum(np.exp(neg_k_dots - max_neg), axis=1))
            soft_min_dots = -log_sum / k
            angles = np.arccos(np.clip(soft_min_dots, -1.0, 1.0))
            
            k2 = 20.0
            max_ang = np.max(angles)
            log_sum_exp = max_ang + np.log(np.sum(np.exp(k2 * (angles - max_ang))))
            return log_sum_exp / k2
        
        result = minimize(
            fine_objective,
            best_pts.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.fun < best_radius:
            opt_pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            opt_pts = opt_pts / norms
            opt_radius = _covering_radius_grid(opt_pts, n_theta=200, n_phi=400)
            if opt_radius < best_radius:
                best_pts = opt_pts
                best_radius = opt_radius
    
    # Ensure valid output
    best_pts = np.asarray(best_pts)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
