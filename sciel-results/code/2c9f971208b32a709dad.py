"""Rewritten program: optimize 50-point spherical covering via differentiable
log-sum-exp surrogate for the true spherical circumradius, using L-BFGS-B
on tangent-space coordinates, with multiple random restarts and a deadline guard."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 50
DEADLINE = 135.0  # safety margin before 150s kill

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _spherical_circumradius(facet_points):
    """Compute the spherical circumradius (in radians) of a spherical triangle
    defined by 3 unit vectors. Returns the angular distance from the facet
    centroid (in the spherical sense) to the farthest vertex."""
    # For a spherical triangle, the circumcenter is the point equidistant from all 3 vertices.
    # The circumradius is the angular distance from circumcenter to any vertex.
    # For a triangle with vertices a, b, c (unit vectors), the circumcenter is proportional to
    # a x (b x c) + b x (c x a) + c x (a x b) ... actually let's use a simpler approach.
    
    # The circumcenter of a spherical triangle can be found as follows:
    # It lies in the plane spanned by the three vertices (the great-circle plane).
    # Actually, the circumcenter is the intersection of the three perpendicular bisectors.
    # 
    # Simpler: for a triangle with side lengths (angular distances) a, b, c,
    # the circumradius R satisfies: sin(R) = (sin(a)*sin(b)*sin(c)) / (2*sin(s-a)*sin(s-b)*sin(s-c))
    # where s = (a+b+c)/2, using spherical law of cosines... 
    # Actually, the formula is: R = arcsin( (sin(A)*sin(B)*sin(C)) / (4*cos(A/2)*cos(B/2)*cos(C/2)) )
    # where A, B, C are the angles of the spherical triangle.
    
    # Let me use a more direct geometric approach.
    # The circumcenter of a spherical triangle is the point on the sphere equidistant from all three vertices.
    # In 3D, it's the intersection of the sphere with the line through the origin in the direction
    # of the vector that is equidistant (in chordal distance) from all three points.
    
    a, b, c = facet_points
    
    # The circumcenter direction is proportional to:
    # (b x c) * |a|^2 + (c x a) * |b|^2 + (a x b) * |c|^2 ... no.
    # Actually for unit vectors, the circumcenter is in the direction of:
    # a*(sin(2*angle_ab) + sin(2*angle_ac) - sin(2*angle_bc)) ... this is getting complex.
    
    # Let me just compute it numerically: the circumcenter is the point on the great circle plane
    # equidistant from all three. Since all points are on the unit sphere, the circumcenter is
    # the normalized sum: (a + b + c) doesn't work in general.
    
    # Better approach: solve for point p = alpha*a + beta*b + gamma*c such that |p-a|=|p-b|=|p-c|
    # and p is on the unit sphere.
    
    # Actually, the simplest robust method: the circumcenter of a spherical triangle
    # with vertices a, b, c (unit) is given by the normalized vector:
    # p = (a * sin(2*theta_bc) + b * sin(2*theta_ca) + c * sin(2*theta_ab))
    # where theta_xy is the angle between x and y... no that's not right either.
    
    # Let me use the fact that the circumcenter is the pole of the plane through a,b,c
    # only for right triangles. For general triangles, let me solve the linear system.
    
    # The circumcenter p satisfies:
    # |p - a|^2 = |p - b|^2 => 2*p.(b-a) = |b|^2 - |a|^2 = 0 (since unit)
    # So p.(b-a) = 0 and p.(c-a) = 0
    # This means p is perpendicular to (b-a) and (c-a), so p is parallel to (b-a) x (c-a)
    # Wait, that's the normal to the plane containing a, b, c. But the circumcenter should be
    # in that plane, not perpendicular to it.
    
    # I made an error. |p-a|^2 = |p|^2 - 2p.a + |a|^2 = 1 - 2p.a + 1 = 2 - 2p.a
    # Similarly |p-b|^2 = 2 - 2p.b
    # Setting equal: p.a = p.b, so p.(b-a) = 0.
    # Similarly p.(c-a) = 0.
    # So p is perpendicular to both (b-a) and (c-a), meaning p is parallel to (b-a) x (c-a).
    # But (b-a) x (c-a) is the normal to the plane through a,b,c.
    # This means the circumcenter is along the normal direction? That can't be right for a
    # spherical triangle unless I'm confusing chordal and spherical distance.
    
    # The issue: I'm using chordal distance equality, but the circumradius in the covering
    # problem is the angular (spherical) distance. The point that minimizes the maximum
    # angular distance to the three vertices is NOT the same as the chordal circumcenter.
    
    # For the covering radius, what matters is: for each facet (spherical triangle), the
    # worst point is the spherical circumcenter (point equidistant in angular distance to all 3
    # vertices), and the covering radius contribution is that angular distance.
    
    # The spherical circumcenter: point p on sphere with angular(p,a) = angular(p,b) = angular(p,c)
    # This means p.a = p.b = p.c (dot products equal), so p.(b-a)=0, p.(c-a)=0.
    # Same equations! So p is parallel to (b-a) x (c-a).
    
    n = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return 0.0
    p = n / n_norm
    
    # The circumradius is the angular distance from p to a (or b or c)
    dot_val = np.clip(np.dot(p, a), -1.0, 1.0)
    R = np.arccos(dot_val)
    
    # The circumcenter could be on the "wrong" side (antipodal). We want the one in the
    # interior of the triangle, which corresponds to the smaller angle.
    if R > np.pi / 2:
        R = np.pi - R  # Actually, we should check: the circumcenter of the facet
                       # is the one that's the "farthest" point in the Voronoi cell.
                       # For a convex hull facet, the relevant circumcenter is the one
                       # on the same side as the facet normal (outward).
    
    return R

def _covering_radius(points):
    """Compute the covering radius in radians."""
    # Compute convex hull of the 50 points
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback
    
    max_R = 0.0
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        R = _spherical_circumradius(facet_pts)
        # The relevant circumcenter is the one that lies in the Voronoi cell.
        # For the convex hull, the outward normal points away from origin.
        # The circumcenter we computed might be on either side.
        # We need the one that is "outside" the convex hull (i.e., in the direction of the facet normal).
        # Actually, for the covering radius, we need the point on the sphere that is farthest
        # from all 50 points. This is the maximum over all Voronoi vertices (which are
        # circumcenters of facets) of the distance to the nearest site.
        
        # The circumcenter computed above: let's check which side it's on.
        a, b, c = facet_pts
        n = np.cross(b - a, c - a)
        # The facet normal for the convex hull (outward)
        # hull.equations gives plane equations
        # For now, use the circumradius as computed; take the min(R, pi-R) to get the
        # distance to the nearest vertex from the circumcenter.
        R_eff = min(R, np.pi - R)
        max_R = max(max_R, R_eff)
    
    return max_R

def _logsumexp_circumradius(points, temperature=0.05):
    """Differentiable surrogate: log-sum-exp of circumradius contributions."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, 0.0
    
    Rs = []
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        R = _spherical_circumradius(facet_pts)
        R_eff = min(R, np.pi - R)
        Rs.append(R_eff)
    
    Rs = np.array(Rs)
    max_R = np.max(Rs)
    # Log-sum-exp surrogate
    shifted = Rs / temperature
    max_shifted = np.max(shifted)
    lse = temperature * (max_shifted + np.log(np.sum(np.exp(shifted - max_shifted))))
    return lse, max_R

def _tangent_to_sphere(tangent, base_point):
    """Map tangent vector to sphere via exponential map."""
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return base_point.copy()
    # Exponential map: p = cos(norm)*base + sin(norm)*tangent/norm
    return np.cos(norm) * base_point + np.sin(norm) * tangent / norm

def _sphere_to_tangent(point, base_point):
    """Map sphere point to tangent space at base_point via logarithmic map."""
    # Project point onto tangent space at base_point
    # tangent = (point - (point.base)*base) / |point - (point.base)*base|
    dot = np.dot(point, base_point)
    dot_clipped = np.clip(dot, -1.0 + 1e-10, 1.0 - 1e-10)
    # For points near the base, use first-order: tangent ≈ point - (point.base)*base
    # For robustness, use the full logarithmic map
    # angle = arccos(dot)
    # tangent_dir = (point - dot*base) / |point - dot*base|
    # tangent = angle * tangent_dir
    v = point - dot * base_point
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros_like(base_point)
    angle = np.arccos(dot_clipped)
    return angle * v / v_norm

def construct():
    start_time = time.time()
    best_points = None
    best_radius = np.pi
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Evaluate initial
    try:
        init_radius = _covering_radius(init_points)
        best_points = init_points.copy()
        best_radius = init_radius
    except Exception:
        pass
    
    # Try multiple random restarts with local optimization
    rng = np.random.default_rng(42)
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            points = init_points.copy()
        else:
            # Random perturbation of initial points
            points = init_points.copy()
            for k in range(N):
                # Random small perturbation
                noise = rng.normal(0, 0.1, 3)
                points[k] = _normalize(points[k] + noise)
        
        # Use tangent-space optimization
        base_point = points[0].copy()
        tangent = np.zeros((N, 3))
        for k in range(N):
            tangent[k] = _sphere_to_tangent(points[k], base_point)
        
        def objective(x):
            if time.time() - start_time > DEADLINE:
                return np.pi
            # Convert tangent coordinates back to sphere
            pts = np.zeros((N, 3))
            for k in range(N):
                pts[k] = _tangent_to_sphere(x[k], base_point)
            # Ensure normalization
            for k in range(N):
                pts[k] = _normalize(pts[k])
            val, _ = _logsumexp_circumradius(pts, temperature=0.03)
            return val
        
        try:
            result = minimize(
                objective,
                tangent,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if result.success or result.fun < best_radius:
                # Convert back to sphere points
                opt_pts = np.zeros((N, 3))
                for k in range(N):
                    opt_pts[k] = _tangent_to_sphere(result.x[k], base_point)
                    opt_pts[k] = _normalize(opt_pts[k])
                
                # Evaluate true covering radius
                true_radius = _covering_radius(opt_pts)
                if true_radius < best_radius:
                    best_radius = true_radius
                    best_points = opt_pts.copy()
        except Exception:
            continue
    
    # Final refinement: a few more L-BFGS-B steps on the best found
    if best_points is not None and time.time() - start_time < DEADLINE:
        base_point = best_points[0].copy()
        tangent = np.zeros((N, 3))
        for k in range(N):
            tangent[k] = _sphere_to_tangent(best_points[k], base_point)
        
        def objective2(x):
            if time.time() - start_time > DEADLINE:
                return np.pi
            pts = np.zeros((N, 3))
            for k in range(N):
                pts[k] = _tangent_to_sphere(x[k], base_point)
                pts[k] = _normalize(pts[k])
            val, _ = _logsumexp_circumradius(pts, temperature=0.01)
            return val
        
        try:
            result = minimize(
                objective2,
                tangent,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 30}
            )
            if result.fun < best_radius:
                opt_pts = np.zeros((N, 3))
                for k in range(N):
                    opt_pts[k] = _tangent_to_sphere(result.x[k], base_point)
                    opt_pts[k] = _normalize(opt_pts[k])
                true_radius = _covering_radius(opt_pts)
                if true_radius < best_radius:
                    best_radius = true_radius
                    best_points = opt_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = init_points
    
    # Final normalization
    for k in range(N):
        best_points[k] = _normalize(best_points[k])
    
    return best_points
