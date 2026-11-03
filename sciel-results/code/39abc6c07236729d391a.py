"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

# EVOLVE-BLOCK-START

def _compute_circumradius(tri):
    """Compute the circumradius of a triangle on the unit sphere.
    Returns the angular distance from the circumcenter to the vertices."""
    a, b, c = tri
    # Normalize
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)
    
    # Side lengths (chord lengths)
    ab = np.linalg.norm(a - b)
    bc = np.linalg.norm(b - c)
    ca = np.linalg.norm(c - a)
    
    # Area of triangle on sphere using vector triple product
    cross_ab = np.cross(a, b)
    area_vec = np.cross(cross_ab, c)
    area = np.linalg.norm(area_vec)
    
    # Circumradius of spherical triangle
    # Using formula: R = abc / (4 * area) for planar, but for spherical we need to be careful
    # Actually, for points on unit sphere, the circumradius in angular measure:
    # The circumcenter is the point equidistant from a, b, c on the sphere
    
    # Find circumcenter: it's the normalized vector (a * sin(A) + b * sin(B) + c * sin(C))
    # where A, B, C are angles opposite to a, b, c
    # Or simpler: circumcenter is in the plane of a,b,c, so it's proportional to a x (b x c)
    # Wait, that's the normal. The circumcenter lies in the plane spanned by a,b,c.
    
    # The circumcenter x satisfies: |x-a| = |x-b| = |x-c| and |x|=1
    # This means x is in the plane of a,b,c and equidistant from all three.
    
    # Let's solve: x = alpha*a + beta*b + gamma*c with constraints
    # Actually, the circumcenter of a spherical triangle can be found as:
    # x = (a * sin(2A) + b * sin(2B) + c * sin(2C)) / |...|
    
    # Simpler approach: The circumcenter is the intersection of the perpendicular bisectors.
    # On the sphere, the perpendicular bisector of arc AB is the great circle through the midpoint of AB and the pole.
    
    # Let's use the formula: circumcenter = normalize(a * (b·c - a·b) + b * (c·a - b·c) + c * (a·b - c·a))
    # No, that's not right either.
    
    # Correct formula: The circumcenter x satisfies (x-a)·(b-c) = 0, (x-b)·(c-a) = 0, (x-c)·(a-b) = 0
    # and |x| = 1.
    
    # From the first two:
    # x·(b-c) = a·(b-c)
    # x·(c-a) = b·(c-a)
    
    # Let u = b-c, v = c-a
    # x·u = a·u
    # x·v = b·v
    
    # x is in the plane of a,b,c, so x = p*a + q*b + r*c
    # But we can solve the linear system for x in the 3D space with the constraint |x|=1
    
    # Actually, the circumcenter lies in the plane containing a,b,c. The plane has normal n = a x (b x c) / |a x (b x c)|
    # Wait, the plane through origin and a,b,c has normal n = (a x b) x c or similar.
    
    # Let's just solve the linear system. The circumcenter x satisfies:
    # 2*x·(b-c) = |b|^2 - |c|^2 = 0 (since all on unit sphere)
    # So x·(b-c) = 0
    # x·(c-a) = 0
    # x·(a-b) = 0
    
    # Wait, that means x is orthogonal to all edge vectors, so x is parallel to the normal of the plane!
    # But the normal of the plane through a,b,c (and origin) is n = a x (b x c)
    
    # So the circumcenter is ±n/|n|. We need to choose the sign such that it's on the same side as the triangle.
    
    n = np.cross(a, np.cross(b, c))
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate triangle
        return np.pi  # Large radius
    
    x = n / n_norm
    
    # Check which sign: the circumcenter should be such that the angle from x to a is the circumradius
    # Compute the angular distance from x to a
    cos_theta = np.clip(np.dot(x, a), -1, 1)
    theta = np.arccos(cos_theta)
    
    # Also check to b and c to verify
    cos_theta_b = np.clip(np.dot(x, b), -1, 1)
    theta_b = np.arccos(cos_theta_b)
    
    cos_theta_c = np.clip(np.dot(x, c), -1, 1)
    theta_c = np.arccos(cos_theta_c)
    
    # The circumradius is the maximum of these (should be equal if x is correct)
    # But we might have the wrong sign
    if abs(theta - theta_b) > 1e-6 or abs(theta - theta_c) > 1e-6:
        # Try the other sign
        x = -x
        cos_theta = np.clip(np.dot(x, a), -1, 1)
        theta = np.arccos(cos_theta)
        cos_theta_b = np.clip(np.dot(x, b), -1, 1)
        theta_b = np.arccos(cos_theta_b)
        cos_theta_c = np.clip(np.dot(x, c), -1, 1)
        theta_c = np.arccos(cos_theta_c)
    
    # The circumradius is the angular distance
    return max(theta, theta_b, theta_c)


def _covering_radius(points):
    """Compute the covering radius of a set of points on the unit sphere."""
    points = np.array(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If convex hull fails, use a fallback
        # Compute max distance to nearest point on a grid
        max_dist = 0
        for i in range(100):
            for j in range(100):
                theta = np.pi * i / 100
                phi = 2 * np.pi * j / 100
                y = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                dists = np.arccos(np.clip(points @ y, -1, 1))
                max_dist = max(max_dist, np.min(dists))
        return np.degrees(max_dist)
    
    max_radius = 0
    for simplex in hull.simplices:
        tri = points[simplex]
        r = _compute_circumradius(tri)
        max_radius = max(max_radius, r)
    
    return np.degrees(max_radius)


def _lloyd_iterate(points, n_iter=10):
    """Run Lloyd's algorithm for spherical Voronoi cells."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    for _ in range(n_iter):
        # Compute Voronoi cell centroids
        # For each point, find all points that are closer to it than to any other point
        # This is expensive, so we'll use a simplified version:
        # Move each point towards the centroid of its neighbors
        
        new_points = np.zeros_like(points)
        for i in range(N):
            # Find points closer to i than to any other point
            # Approximate: find the 10 nearest neighbors and move towards their centroid
            dists = np.array([np.linalg.norm(points[i] - points[j]) for j in range(N)])
            # Exclude self
            dists[i] = np.inf
            # Get 10 nearest
            neighbors = np.argsort(dists)[:10]
            # Weighted centroid
            weights = 1.0 / (dists[neighbors] + 1e-6)
            centroid = np.sum(points[neighbors] * weights[:, None], axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c > 1e-6:
                new_points[i] = centroid / norm_c
            else:
                new_points[i] = points[i]
        
        points = new_points
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    return points


def _electrostatic_refine(points, n_iter=50):
    """Refine points using electrostatic repulsion."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    for _ in range(n_iter):
        # Compute repulsive forces
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                # Repulsive force proportional to 1/dist^2
                force = diff / (dist ** 3)
                forces[i] += force
        
        # Update points
        step = 0.01
        points = points + step * forces
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return points


def _smooth_objective(x, points_flat):
    """Smooth surrogate for the covering radius using log-sum-exp."""
    points = x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Compute circumradii for all facets
    try:
        hull = ConvexHull(points)
    except Exception:
        return 100.0  # Large penalty
    
    radii = []
    for simplex in hull.simplices:
        tri = points[simplex]
        r = _compute_circumradius(tri)
        radii.append(r ** 2)
    
    radii = np.array(radii)
    if len(radii) == 0:
        return 100.0
    
    # Log-sum-exp for smooth max
    temp = 0.01
    max_r = np.max(radii)
    log_sum_exp = max_r + temp * np.log(np.sum(np.exp((radii - max_r) / temp)))
    return log_sum_exp


def _gradient_smooth(x, points_flat):
    """Gradient of the smooth objective (numerical)."""
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = _smooth_objective(x, points_flat)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _smooth_objective(x_plus, points_flat)
        grad[i] = (f_plus - f0) / eps
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    best_points = None
    best_radius = np.inf
    
    # Generate multiple initial configurations
    initializations = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib)
    
    # 2. Random + electrostatic
    rng = np.random.default_rng(42)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        pts = _electrostatic_refine(pts, n_iter=20)
        initializations.append(pts)
    
    # 3. Another Fibonacci variant
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.1)
    r = np.sqrt(1.0 - z * z)
    fib2 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib2)
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        # Lloyd's algorithm
        pts = _lloyd_iterate(init, n_iter=5)
        radius = _covering_radius(pts)
        
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()
        
        # Gradient descent with smooth surrogate
        if time.time() > deadline - 20:
            break
        
        x = pts.flatten()
        try:
            result = minimize(
                _smooth_objective,
                x,
                args=(x,),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.success:
                pts_grad = result.x.reshape(N, 3)
                norms = np.linalg.norm(pts_grad, axis=1, keepdims=True)
                pts_grad = pts_grad / norms
                radius_grad = _covering_radius(pts_grad)
                if radius_grad < best_radius:
                    best_radius = radius_grad
                    best_points = pts_grad.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead if time allows
    if best_points is not None and time.time() < deadline - 10:
        x = best_points.flatten()
        try:
            result = minimize(
                _smooth_objective,
                x,
                args=(x,),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.success:
                pts_polish = result.x.reshape(N, 3)
                norms = np.linalg.norm(pts_polish, axis=1, keepdims=True)
                pts_polish = pts_polish / norms
                radius_polish = _covering_radius(pts_polish)
                if radius_polish < best_radius:
                    best_radius = radius_polish
                    best_points = pts_polish.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
