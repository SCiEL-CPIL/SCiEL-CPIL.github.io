"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0  # wall-clock limit
SAFETY_MARGIN = 10.0  # seconds before limit to stop

# EVOLVE-BLOCK-START


def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Returns the max angular distance from any point on sphere to nearest point.
    Uses convex hull facets: for each facet, compute circumradius.
    The covering radius is the max circumradius of all facets.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the 3 points of the triangle
            tri = points[simplex]
            # Compute circumradius of the triangle on the unit sphere
            # For a spherical triangle with vertices a, b, c (unit vectors),
            # the circumradius is the angular distance from the circumcenter to any vertex.
            # The circumcenter of a spherical triangle can be found via:
            # n = (a x b) + (b x c) + (c x a)  ... no, that's not right.
            
            # Actually, for a triangle on the unit sphere with vertices a, b, c,
            # the circumradius R satisfies: sin(R) = |a x b| / (2 * d) where d is... 
            # Let me use a different approach.
            
            # The circumradius of a spherical triangle can be computed as:
            # Find the point p on the sphere equidistant from a, b, c.
            # p · a = p · b = p · c = cos(R)
            # This means p is in the span of a, b, c and is perpendicular to (a-b), (b-c).
            
            # Alternative: use the formula for circumradius of a triangle in 3D projected to sphere.
            # For a planar triangle with side lengths a_s, b_s, c_s (chord lengths),
            # and circumradius R_planar, the spherical circumradius is:
            # R_sphere = arcsin(R_planar / 2) ... no that's not right either.
            
            # Let me just compute it directly:
            # The circumcenter of a triangle with vertices a, b, c (unit vectors) 
            # in 3D (planar circumcenter) projected onto the sphere.
            
            # Planar circumcenter of triangle with vertices a, b, c:
            # Solve for p = alpha*a + beta*b + gamma*c such that |p-a|^2 = |p-b|^2 = |p-c|^2
            # and p is in the plane of a, b, c.
            
            # Actually, simpler: the circumradius of a triangle inscribed in a unit sphere
            # can be found by: R = arcsin(|a x b| / (2 * |a - b|)) ... no.
            
            # Let me use the direct approach:
            # The circumcenter of triangle (a,b,c) in 3D space:
            # Let A = a, B = b, C = c
            # The circumcenter O satisfies |O-A|^2 = |O-B|^2 = |O-C|^2
            # This gives: 2*O·(B-A) = |B|^2 - |A|^2 = 0 (since unit vectors)
            #            2*O·(C-A) = |C|^2 - |A|^2 = 0
            # So O is perpendicular to (B-A) and (C-A), meaning O is parallel to (B-A) x (C-A)
            
            n = np.cross(B - A, C - A) if False else np.cross(b - a, c - a)
            # Wait, I need to be careful. Let me redo.
            
            a = tri[0]
            b = tri[1]
            c = tri[2]
            
            # Circumcenter of planar triangle with vertices a, b, c
            # O = t1 * (b - a) + t2 * (c - a) ... no
            # The circumcenter is along the normal to the plane through a, b, c
            # that passes through the origin... no, the circumcenter is in the plane.
            
            # For unit vectors a, b, c, the planar circumcenter O satisfies:
            # O · (b - a) = 0 and O · (c - a) = 0 (since |O-a|^2 = |O-b|^2 => 2O·(b-a) = |b|^2-|a|^2 = 0)
            # So O is parallel to n = (b-a) x (c-a)
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # O = s * n for some scalar s
            # |O - a|^2 = |s*n - a|^2 = s^2 - 2*s*(n·a) + |a|^2 = s^2 - 2*s*(n·a) + 1
            # We want the circumradius to be |O - a| (planar distance), and the spherical 
            # circumradius is the angle between a and O/|O| (if O is on the sphere side)
            # Actually, the circumcenter on the sphere is the point p on the sphere such that
            # p·a = p·b = p·c. Since p is in the span of a,b,c and equidistant...
            
            # Let me think differently. The spherical circumradius R is defined such that
            # there exists a point p on the unit sphere with angle(p,a) = angle(p,b) = angle(p,c) = R.
            # So p·a = p·b = p·c = cos(R).
            # p is in the plane of a,b,c (the great circle plane through them? No, the plane containing a,b,c).
            # p = alpha*a + beta*b + gamma*c
            # p·a = p·b = p·c => (alpha*a+beta*b+gamma*c)·a = (alpha*a+beta*b+gamma*c)·b = ...
            # This is getting complex. Let me just use the formula:
            # For a spherical triangle, circumradius R satisfies:
            # sin(R) = (sin(A) * sin(B) * sin(C)) / (2 * sin(A) * sin(B) * sin(C) ...) 
            # No, let me use the chord-length approach.
            
            # Chord lengths:
            d_ab = np.linalg.norm(a - b)
            d_bc = np.linalg.norm(b - c)
            d_ca = np.linalg.norm(c - a)
            
            # Area of planar triangle with vertices a, b, c:
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            
            # Circumradius of planar triangle: R_planar = (d_ab * d_bc * d_ca) / (4 * area)
            if area < 1e-12:
                continue
            R_planar = (d_ab * d_bc * d_ca) / (4.0 * area)
            
            # The spherical circumradius: the circumcenter on the sphere is at distance R_planar
            # from each vertex in 3D (chord distance), so the angular distance is arcsin(R_planar / 2)
            # Wait, that's for a circle of radius R_planar in 3D. The chord distance from circumcenter 
            # to vertex is R_planar. If the circumcenter is on the unit sphere, then the chord distance
            # from circumcenter to vertex is 2*sin(R/2) where R is the angular distance.
            # So R_planar = 2*sin(R/2), thus R = 2*arcsin(R_planar/2)
            
            if R_planar > 1.0:
                # This can happen for obtuse triangles; the circumcenter is outside the unit sphere
                # In that case, the farthest point on the sphere from the triangle is one of the vertices
                # or the midpoint of an edge. Let's just use the max angular distance to vertices.
                # Actually, for the covering radius, what matters is the farthest point on the sphere
                # from the triangle. For an obtuse spherical triangle, this is the vertex opposite 
                # the longest side.
                R = 2.0 * np.arcsin(min(R_planar, 1.0) / 2.0)
            else:
                R = 2.0 * np.arcsin(R_planar / 2.0)
            
            max_r = max(max_r, R)
        
        return np.degrees(max_r)
    except Exception:
        # Fallback: compute max distance to nearest point on a fine grid
        # This is expensive but safe
        try:
            # Use a coarse grid
            theta = np.linspace(0, np.pi, 30)
            phi = np.linspace(0, 2*np.pi, 60)
            TH, PH = np.meshgrid(theta, phi)
            Y = np.column_stack([
                np.sin(TH) * np.cos(PH).ravel(),
                np.sin(TH) * np.sin(PH).ravel(),
                np.cos(TH).ravel()
            ])
            # Compute angular distance from each Y to nearest point
            dots = Y @ points.T  # shape (nY, N)
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            min_angles = angles.min(axis=1)
            return np.degrees(min_angles.max())
        except Exception:
            return 180.0


def _soft_covering_radius(points, beta=50.0):
    """
    Soft version of covering radius using log-sum-exp for smoothness.
    This is a surrogate for the nonsmooth max operation.
    """
    # Use a grid of test points
    n_theta = 20
    n_phi = 40
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    
    # Angular distances
    dots = Y @ points.T  # (nY, N)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    
    # Soft max over the grid points
    # log-sum-exp trick
    a = angles.ravel()
    a_max = a.max()
    if a_max > 1e-10:
        soft_max = a_max + np.log(np.sum(np.exp(beta * (a - a_max)))) / beta
    else:
        soft_max = 0.0
    
    return soft_max


def _repulsive_energy(points, k=1.0):
    """
    Repulsive energy: sum over pairs of 1/d^2 where d is angular distance.
    This encourages points to spread out.
    """
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    np.fill_diagonal(angles, 1e10)
    # Use a softened inverse square to avoid singularities
    d = np.maximum(angles, 0.01)
    energy = np.sum(1.0 / (d ** 2))
    return energy


def _pressure_energy(points, beta=50.0):
    """
    Pressure energy: pushes points away from worst-case directions.
    Uses the soft covering radius as a proxy.
    """
    return _soft_covering_radius(points, beta)


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = _covering_radius(best_points)
    
    # Try multiple random initializations and local optimizations
    rng = np.random.default_rng(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        if restart == 0:
            # Use the Fibonacci spiral
            pts = points.copy()
        else:
            # Random initialization
            pts = rng.standard_normal((N, 3))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
        
        # Local optimization using projected gradient descent
        # Objective: minimize covering radius (use soft version for gradient)
        
        def objective_and_grad(pts):
            # Soft covering radius + repulsive term
            soft_cr = _soft_covering_radius(pts, beta=30.0)
            
            # Repulsive energy gradient
            dots = pts @ pts.T
            np.fill_diagonal(dots, 0.0)
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            np.fill_diagonal(angles, 1e10)
            d = np.maximum(angles, 0.01)
            
            # Gradient of sum 1/d^2 with respect to each point
            # d(1/d^2)/d(pts[i]) = -2/d^3 * d(pts[i])/d(angle)
            # This is complex; use numerical gradient or skip
            # For now, just use the soft covering radius gradient numerically
            
            return soft_cr
        
        # Use L-BFGS-B with projection
        def project_to_sphere(pts):
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            return pts / norms
        
        # Momentum-based projected gradient descent
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(pts)
        n_iters = 200
        
        for it in range(n_iters):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute gradient numerically
            eps = 1e-5
            grad = np.zeros_like(pts)
            base_val = _soft_covering_radius(pts, beta=30.0)
            
            # Only do a few coordinate updates per iteration for speed
            # Use random subset
            if N <= 50:
                indices = np.arange(N)
            else:
                indices = rng.choice(N, size=min(10, N), replace=False)
            
            for idx in indices:
                for dim in range(3):
                    pts_p = pts.copy()
                    pts_p[idx, dim] += eps
                    pts_p = project_to_sphere(pts_p)
                    val_p = _soft_covering_radius(pts_p, beta=30.0)
                    
                    pts_m = pts.copy()
                    pts_m[idx, dim] -= eps
                    pts_m = project_to_sphere(pts_m)
                    val_m = _soft_covering_radius(pts_m, beta=30.0)
                    
                    grad[idx, dim] = (val_p - val_m) / (2 * eps)
            
            # Update with momentum
            v = momentum * v - lr * grad
            pts = pts + v
            pts = project_to_sphere(pts)
        
        # Evaluate final score
        score = _covering_radius(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    return best_points


# EVOLVE-BLOCK-END
