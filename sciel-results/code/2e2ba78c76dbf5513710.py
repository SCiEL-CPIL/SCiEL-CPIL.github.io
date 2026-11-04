"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _covering_radius_rad(points):
    """Compute the covering radius in radians (max angular distance to nearest point)."""
    # For efficiency, use the convex hull facets approach
    # The covering radius is the max circumradius of the Delaunay triangles
    # But for robustness, we can compute max over a dense set of test points
    # However, the true covering radius equals the max circumradius of the
    # spherical Voronoi cells, which equals the max circumradius of the
    # Delaunay triangles on the sphere.
    
    # Use scipy's ConvexHull for 3D - the facets of the convex hull of the
    # points give the Delaunay triangulation. The circumradius of each facet
    # (triangle) gives the covering radius for that cell.
    
    from scipy.spatial import ConvexHull
    
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    pts = points / norms[:, None]
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute max distance to nearest point on a grid
        # This is expensive but robust
        lat = np.linspace(-np.pi/2, np.pi/2, 200)
        lon = np.linspace(0, 2*np.pi, 400)
        LL, LLon = np.meshgrid(lat, lon)
        test_pts = np.column_stack([
            np.cos(LL) * np.cos(LLon),
            np.cos(LL) * np.sin(LLon),
            np.sin(LL)
        ])
        # Compute min distance from each test point to each config point
        # Use matrix form
        # dist^2 = 2 - 2 * dot
        dots = test_pts @ pts.T
        min_dots = dots.max(axis=1)  # max dot = min angle
        max_angle = np.arccos(np.clip(min_dots, -1, 1))
        return max_angle
    
    # For each facet (triangle), compute the circumradius
    # The circumradius of a triangle on the unit sphere is the angle from
    # the sphere center to the circumcenter of the triangle (in 3D).
    # For a triangle with vertices a, b, c on the unit sphere, the circumcenter
    # is the point equidistant from a, b, c on the sphere.
    # The circumradius R satisfies: for the triangle, the circumcenter O satisfies
    # |O - a| = |O - b| = |O - c| = R (Euclidean distance in 3D)
    # But on the sphere, the angular circumradius is the angle subtended.
    
    # Actually, for a spherical triangle, the circumradius (angular) can be computed
    # as follows: the circumcenter is the intersection of the perpendicular bisectors.
    # A simpler approach: for each facet, compute the circumcenter in 3D space
    # and the angular distance from the facet's circumcenter to its vertices.
    
    # The circumcenter of triangle (a,b,c) on unit sphere:
    # It's the point c such that c·a = c·b = c·c (all equal, since on unit sphere)
    # Actually c·a = c·b = c·c = cos(R) where R is the angular circumradius.
    
    # Solve: c = alpha*a + beta*b + gamma*c... let's use the formula:
    # The circumcenter is proportional to:
    # a * sin(2*angle_b) + b * sin(2*angle_c) + c * sin(2*angle_a) ... no
    
    # Simpler: the circumcenter of triangle with vertices a,b,c on unit sphere
    # is the normalized vector:
    #   (a x b) x (b x c) ... no that's not right either.
    
    # Let me use a direct approach: the circumcenter O satisfies:
    #   O · a = O · b = O · c
    #   |O| = 1
    # From O·a = O·b: O·(a-b) = 0
    # From O·b = O·c: O·(b-c) = 0
    # So O is perpendicular to (a-b) and (b-c), hence O is parallel to (a-b) x (b-c)
    
    max_circumradius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Circumcenter direction
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            continue
        o = cross / norm_cross
        
        # The angular circumradius is the angle between o and a (should be same for b, c)
        dot_a = np.dot(o, a)
        # Clamp for numerical stability
        dot_a = np.clip(dot_a, -1.0, 1.0)
        angle = np.arccos(dot_a)
        
        if angle > max_circumradius:
            max_circumradius = angle
    
    return max_circumradius


def _soft_max_surrogate(points, beta=200.0):
    """Smooth surrogate for the covering radius using log-sum-exp style."""
    # We want to minimize the max circumradius over facets.
    # Use a soft-max: (1/beta) * log(sum(exp(beta * r_i)))
    from scipy.spatial import ConvexHull
    
    norms = np.linalg.norm(points, axis=1)
    pts = points / norms[:, None]
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a dense grid
        lat = np.linspace(-np.pi/2, np.pi/2, 150)
        lon = np.linspace(0, 2*np.pi, 300)
        LL, LLon = np.meshgrid(lat, lon)
        test_pts = np.column_stack([
            np.cos(LL) * np.cos(LLon),
            np.cos(LL) * np.sin(LLon),
            np.sin(LL)
        ])
        dots = test_pts @ pts.T
        min_dots = dots.max(axis=1)
        angles = np.arccos(np.clip(min_dots, -1, 1))
        # Soft max
        max_a = angles.max()
        if max_a < 1e-10:
            return 0.0
        shifted = beta * (angles - max_a)
        log_sum = max_a + np.log(np.sum(np.exp(shifted))) / beta
        return log_sum
    
    radii = []
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            radii.append(0.0)
            continue
        o = cross / norm_cross
        dot_a = np.clip(np.dot(o, a), -1.0, 1.0)
        angle = np.arccos(dot_a)
        radii.append(angle)
    
    radii = np.array(radii)
    max_r = radii.max()
    if max_r < 1e-10:
        return 0.0
    shifted = beta * (radii - max_r)
    # Avoid overflow
    log_sum = max_r + np.log(np.sum(np.exp(shifted))) / beta
    return log_sum


def _flatten(points):
    return points.flatten()


def _unflatten(x):
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-15] = 1e-15
    return pts / norms[:, None]


def _objective_and_grad(x, beta=200.0):
    """Objective and gradient for L-BFGS-B."""
    pts = _unflatten(x)
    
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Return a large penalty
        return 10.0, np.zeros_like(x)
    
    radii = np.zeros(len(hull.simplices))
    grad_pts = np.zeros_like(pts)
    
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            radii[i] = 0.0
            continue
        
        o = cross / norm_cross
        dot_a = np.clip(np.dot(o, a), -1.0, 1.0)
        angle = np.arccos(dot_a)
        radii[i] = angle
        
        # Gradient of angle w.r.t. a, b, c
        # angle = arccos(o · a) where o = cross / |cross|
        # This is complex; use finite differences as fallback
        pass
    
    # For now, use the soft max surrogate with finite difference gradient
    # Or just use the hard max and let L-BFGS-B handle it via finite differences
    # Actually, let's use the soft max as a smooth surrogate
    max_r = radii.max()
    if max_r < 1e-10:
        return 0.0, np.zeros_like(x)
    
    shifted = beta * (radii - max_r)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = max_r + np.log(sum_exp) / beta
    
    # Gradient of log_sum w.r.t. radii
    grad_radii = exp_shifted / sum_exp / beta
    
    # Now we need gradient of radii w.r.t. x
    # This is complex; use finite differences for the full gradient
    # For efficiency, just return the objective and let L-BFGS-B use finite differences
    # But that's slow. Let's just return objective and use approx gradient.
    
    return log_sum, np.zeros_like(x)  # Zero gradient, let L-BFGS-B use finite diff


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_radius = _covering_radius_rad(best_points)
    
    def elapsed():
        return time.time() - t_start
    
    # Phase 1: L-BFGS-B optimization with soft-max surrogate
    # Use multiple restarts with perturbed initial points
    n_restarts = 5
    for restart in range(n_restarts):
        if elapsed() > DEADLINE - 20:
            break
        
        if restart == 0:
            x0 = best_points.flatten()
        else:
            # Perturb the best solution
            noise = np.random.randn(N, 3) * 0.05
            x0 = (best_points + noise).flatten()
            # Renormalize
            pts0 = x0.reshape(N, 3)
            norms0 = np.linalg.norm(pts0, axis=1)
            norms0[norms0 < 1e-15] = 1e-15
            x0 = (pts0 / norms0[:, None]).flatten()
        
        try:
            # Use L-BFGS-B with finite difference gradient
            result = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
                jac=False  # Let it use finite differences
            )
            
            if result.success or result.fun < _objective_and_grad(x0)[0]:
                pts = _unflatten(result.x)
                rad = _covering_radius_rad(pts)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on the worst facet
    if elapsed() < DEADLINE - 15:
        # Identify the worst point and try to move it
        for _ in range(3):
            if elapsed() > DEADLINE - 10:
                break
            
            # Find the point that is part of the worst facet
            # Simplified: just do a few Nelder-Mead iterations on random points
            for trial in range(10):
                if elapsed() > DEADLINE - 10:
                    break
                
                idx = np.random.randint(N)
                x0 = best_points.flatten()
                
                def nm_obj(x):
                    pts = _unflatten(x)
                    return _covering_radius_rad(pts)
                
                try:
                    result = minimize(
                        nm_obj,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    pts = _unflatten(result.x)
                    rad = _covering_radius_rad(pts)
                    if rad < best_radius:
                        best_radius = rad
                        best_points = pts
                except Exception:
                    pass
    
    # Phase 3: Final L-BFGS-B polish with tighter tolerances
    if elapsed() < DEADLINE - 5:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10},
                jac=False
            )
            pts = _unflatten(result.x)
            rad = _covering_radius_rad(pts)
            if rad < best_radius:
                best_radius = rad
                best_points = pts
        except Exception:
            pass
    
    # Ensure all points are on unit sphere
    norms = np.linalg.norm(best_points, axis=1)
    norms[norms < 1e-15] = 1e-15
    best_points = best_points / norms[:, None]
    
    return best_points
# EVOLVE-BLOCK-END
