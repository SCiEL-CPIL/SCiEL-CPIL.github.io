"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
import math

N = 50
DEADLINE = 115.0  # safety margin before 120s kill
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    try:
        hull = ConvexHull(pts)
        # For each facet (simplex), compute the circumradius
        max_r = 0.0
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            # The circumcenter of a simplex on the unit sphere:
            # For a facet with vertices v1,...,vk, the circumradius is the angular
            # distance from the origin's projection to the farthest vertex.
            # Actually, for a spherical cap covering a facet, the covering radius
            # for that facet is the circumradius of the spherical polygon.
            # For a spherical triangle with vertices on unit sphere, the circumradius
            # can be computed as follows:
            # The circumcenter c is the point on the sphere equidistant from all vertices.
            # For a triangle with side lengths a,b,c (angular distances), the circumradius R
            # satisfies: sin(R) = (sin(a)*sin(b)*sin(c)) / (2*sin(a+b/2)*sin(b+c/2)*sin(c+a/2))
            # This is complex. Alternative: compute circumcenter in 3D.
            
            # For a spherical simplex, the covering radius is the maximum angular distance
            # from the circumcenter to any vertex of the facet.
            # The circumcenter of a set of points on the unit sphere is the intersection
            # of the sphere with the line through the origin and the circumcenter of the
            # planar polygon formed by the vertices.
            
            # For a triangle (3 points), the circumcenter in 3D:
            k = len(simplex)
            if k == 3:
                v1, v2, v3 = facet_pts
                # Solve for circumcenter c on unit sphere equidistant from v1,v2,v3
                # c · v1 = c · v2 = c · v3 = cos(R)
                # and ||c|| = 1
                # This gives: c · (v1-v2) = 0, c · (v1-v3) = 0
                # So c is parallel to (v1-v2) × (v1-v3)
                n = np.cross(v1 - v2, v1 - v3)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    # Degenerate
                    r = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                    max_r = max(max_r, r)
                    continue
                c = n / n_norm
                # Check which side: c should be on the same side as the origin for the facet
                # Actually, the circumcenter should be such that it's the center of the cap
                # containing the facet. We need c to be in the direction away from the interior.
                # The facet normal pointing outward from the convex hull:
                # For the convex hull, the outward normal of the facet
                # The circumcenter of the spherical cap covering the facet is along the
                # outward normal direction.
                # Let's compute the circumradius as the angular distance from c to v1
                dot_val = np.dot(c, v1)
                # We need the correct sign: c should point outward from the hull
                # The centroid of the facet
                centroid = np.mean(facet_pts, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    centroid = centroid / centroid_norm
                    if np.dot(c, centroid) < 0:
                        c = -c
                        dot_val = -dot_val
                R = np.arccos(np.clip(dot_val, -1, 1))
                max_r = max(max_r, R)
            elif k == 2:
                # Edge: the covering radius is half the angular distance
                ang = np.arccos(np.clip(np.dot(facet_pts[0], facet_pts[1]), -1, 1))
                max_r = max(max_r, ang / 2.0)
            else:
                # Higher dimensional simplex - unlikely for 3D
                # Use maximum distance from centroid
                centroid = np.mean(facet_pts, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    centroid = centroid / centroid_norm
                for v in facet_pts:
                    ang = np.arccos(np.clip(np.dot(centroid, v), -1, 1))
                    max_r = max(max_r, ang)
        return max_r
    except Exception:
        return np.pi  # fallback: worst case


def _objective_and_grad(params):
    """
    params: flat array of 3*N coordinates
    Returns: (objective_value, gradient)
    Uses log-sum-exp soft-max surrogate for the max covering radius.
    """
    pts = _normalize(params.reshape(N, 3))
    
    # Compute pairwise angular distances
    # For each point on a fine grid, find nearest neighbor distance
    # But that's expensive. Instead, use the convex hull circumradius approach
    # for the true objective, but for gradient we need a smooth surrogate.
    
    # Smooth surrogate: use the sum of exp(beta * d_i) where d_i are the
    # circumradii of facets. But we need differentiability.
    
    # Alternative approach: parameterize points and use a repulsion-based objective
    # that correlates with covering radius.
    
    # For gradient computation, use a differentiable proxy:
    # Minimize the sum of exp(beta * angular_distance_to_nearest) over a set of test points
    # But that's also expensive.
    
    # Let's use a simpler approach: minimize the maximum of exp(beta * R_i) over facets
    # where R_i is the circumradius of facet i.
    
    # For now, use a repulsion energy that encourages uniform distribution
    # This is a surrogate that works well in practice.
    
    # Pairwise repulsion: sum of 1/d_ij^2 or similar
    # But we want to minimize the covering radius, not maximize minimum distance.
    
    # Better: use the fact that for good coverings, points are roughly uniformly
    # distributed. Use a combination of:
    # 1. Repulsion to prevent clustering
    # 2. Attraction to maintain coverage
    
    # Simple approach: minimize sum of exp(beta * d_max_i) where d_max_i is the
    # distance from point i to its farthest neighbor in the Delaunay triangulation.
    
    # For now, let's just compute the covering radius and use finite differences
    # for the gradient, or use a differentiable surrogate.
    
    # Differentiable surrogate: 
    # For each pair (i,j), compute angular distance d_ij
    # The covering radius is related to the maximum over all pairs of d_ij/2
    # (for the midpoint) but that's not quite right.
    
    # Let's use: objective = (1/beta) * log(sum_i exp(beta * r_i))
    # where r_i is the distance from point i to the farthest point in its
    # Voronoi cell. But computing Voronoi cells is hard.
    
    # Practical approach: use the sum of squared distances to nearest neighbor
    # as a proxy, plus a regularization term.
    
    # Actually, let's just compute the covering radius directly and use
    # numerical gradients. It's slower but more accurate.
    
    # For speed, let's use a simpler differentiable objective:
    # Minimize the maximum angular distance between any two adjacent points
    # (in the Delaunay sense), which is a lower bound on the covering radius.
    
    # Simplest differentiable surrogate that works:
    # E = sum_{i<j} exp(-alpha * d_ij)  (repulsion)
    # This encourages points to spread out.
    
    alpha = 10.0
    pts_arr = pts  # (N, 3)
    
    # Compute all pairwise dot products
    dots = np.dot(pts_arr, pts_arr.T)
    np.fill_diagonal(dots, 1.0)
    
    # Angular distances
    d = np.arccos(np.clip(dots, -1.0, 1.0))
    
    # Repulsion energy: sum of exp(-alpha * d) for all pairs
    # This is differentiable
    mask = ~np.eye(N, dtype=bool)
    E = np.sum(np.exp(-alpha * d[mask]))
    
    # Gradient of E with respect to params
    # dE/dd_ij = -alpha * exp(-alpha * d_ij)
    # dd_ij/dv_i = -v_i + (v_i·v_j)/d_ij * v_j  (for unit vectors)
    # This is complex. Let's use finite differences for the gradient.
    
    # Actually, let's use a simpler gradient computation.
    # For unit vectors, the gradient of d_ij with respect to v_i is:
    # grad_{v_i} d_ij = -(v_i - cos(d_ij)*v_j) / sin(d_ij)  (projected to tangent space)
    
    # For now, let's just return the objective and use scipy's numerical gradient
    # or use a simpler formulation.
    
    # Let's use: minimize the sum of exp(beta * max_dist_i) where max_dist_i
    # is the maximum distance from point i to any other point. This encourages
    # uniform distribution.
    
    # Simpler: just minimize the sum of squared pairwise distances to the
    # optimal packing distance.
    
    # For now, let's just return the covering radius and let scipy handle it
    # with numerical gradients. It'll be slower but more accurate.
    
    cr = _covering_radius(pts)
    return cr, None  # None gradient means scipy will use numerical


def _optimize_from_start(start_pts, max_time=30.0):
    """Run L-BFGS-B optimization from a given starting point."""
    start_time = time.time()
    x0 = start_pts.flatten()
    
    best_pts = start_pts.copy()
    best_cr = _covering_radius(start_pts)
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10},
            timeout=max_time
        )
        
        if result.success or result.fun < best_cr:
            new_pts = _normalize(result.x.reshape(N, 3))
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_pts = new_pts
                best_cr = new_cr
    except Exception:
        pass
    
    # Polish with tighter tolerance if time remains
    elapsed = time.time() - start_time
    if elapsed < max_time - 5:
        try:
            x1 = best_pts.flatten()
            result2 = minimize(
                _objective_and_grad,
                x1,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12},
                timeout=max(1, max_time - elapsed - 2)
            )
            if result2.fun < best_cr:
                new_pts = _normalize(result2.x.reshape(N, 3))
                new_cr = _covering_radius(new_pts)
                if new_cr < best_cr:
                    best_pts = new_pts
                    best_cr = new_cr
        except Exception:
            pass
    
    return best_pts, best_cr


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _icosahedron_subdivide():
    """Generate points by subdividing an icosahedron."""
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # Subdivide each face
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each triangle into 4 by adding midpoints
    new_pts = list(verts)
    new_faces = []
    mid_cache = {}
    
    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in mid_cache:
            return mid_cache[key]
        mid = _normalize((new_pts[i] + new_pts[j]) / 2.0)
        idx = len(new_pts)
        new_pts.append(mid)
        mid_cache[key] = idx
        return idx
    
    for face in faces:
        a, b, c = face
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([ab, b, bc])
        new_faces.append([ca, bc, c])
        new_faces.append([ab, bc, ca])
    
    pts = np.array(new_pts)
    pts = _normalize(pts)
    
    # We have 12 + 30 = 42 points. Need 50.
    # Add 8 more points by further subdividing some faces
    # Or just use the 42 and add 8 random points near the "gaps"
    if len(pts) < N:
        # Add random points
        rng = np.random.RandomState(42)
        extra = rng.randn(N - len(pts), 3)
        extra = _normalize(extra)
        pts = np.vstack([pts, extra])
    elif len(pts) > N:
        # Take first N
        pts = pts[:N]
    
    return pts[:N]


def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron subdivision
    try:
        starts.append(_icosahedron_subdivide())
    except Exception:
        pass
    
    # 3. Random starts with different seeds
    for seed in [1, 42, 123, 999]:
        if _time_left() < 20:
            break
        starts.append(_random_sphere(N, seed))
    
    # Optimize from each start
    best_pts = starts[0]
    best_cr = _covering_radius(best_pts)
    
    time_per_start = min(25.0, _time_left() / max(1, len(starts)))
    
    for i, start in enumerate(starts):
        if _time_left() < 10:
            break
        remaining = _time_left() - 5
        alloc = min(time_per_start, remaining / (len(starts) - i))
        if alloc < 5:
            break
        pts, cr = _optimize_from_start(start, max_time=alloc)
        if cr < best_cr:
            best_pts = pts
            best_cr = cr
    
    # Final polish with remaining time
    if _time_left() > 10:
        pts, cr = _optimize_from_start(best_pts, max_time=_time_left() - 5)
        if cr < best_cr:
            best_pts = pts
            best_cr = cr
    
    return _normalize(best_pts)


# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
