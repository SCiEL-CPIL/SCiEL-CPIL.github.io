import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n, offset=0.5):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _facet_data(pts):
    """
    Compute facet circumcenters and their angular radii.
    Returns list of (radius_radians, circumcenter, simplex_indices).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    
    results = []
    for simplex in hull.simplices:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        n = np.cross(edge1, edge2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        
        c = n / norm_n
        # Ensure normal points outward (away from origin, roughly)
        # For a convex hull containing the origin, the outward normal should have positive dot product with the centroid of the facet
        # Actually, for a sphere, the normal should point away from the center (origin).
        # Let's check dot product with the centroid of the triangle.
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        
        # Check if circumcenter is inside the triangle (valid for covering radius calculation as the worst point)
        # If the circumcenter is outside, the worst point is a vertex.
        # We need to distinguish these cases for the objective.
        
        d = np.linalg.norm(c - v1)
        radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
        
        # Check if c is inside the spherical triangle
        # Project v1, v2, v3 onto the plane of the triangle and check if c is inside
        # A simpler check: the angles between c and v1, v2, v3 should all be <= radius
        # And the sum of angles should be related to the area?
        # Actually, for a spherical triangle, the circumcenter is inside if and only if
        # the triangle is acute (all angles < 90 deg in the spherical sense? No, that's Euclidean).
        # Let's use the property that the circumcenter is inside if the dot product of c with the cross product of edges is consistent.
        # A robust check: c is inside if it can be written as a positive combination of v1, v2, v3 on the sphere? No.
        # Let's check if the distance from c to each vertex is the radius, and if c is on the same side of the edges as the interior.
        
        # For now, let's assume if the radius is small, it's likely valid.
        # A better check: The circumcenter is inside the spherical triangle if and only if
        # the angles at the vertices are all less than 90 degrees? No.
        # Let's use the fact that for a valid circumcenter (inside), the dot products c.v1, c.v2, c.v3 are all equal to cos(radius).
        # If the circumcenter is outside, one of the vertices is the furthest, so the covering radius for that facet is the max angle from vertex to nearest point.
        
        dots = np.clip(pts[simplex] @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # If the angles are not all approximately equal, the circumcenter is outside the triangle (or on the boundary)
        if np.max(angles) - np.min(angles) > 1e-5:
            # Circumcenter is outside. The worst point is a vertex.
            # The covering radius for this facet is the max over vertices of the angle to the nearest point in the whole config.
            # But for the objective, we can approximate by using the max angle from the vertex to the nearest of the 3 vertices? No, to the nearest in the whole config.
            # To keep the objective smooth and local, let's just use the max angle from the vertex to the nearest of the 3 vertices as a lower bound?
            # Actually, if the circumcenter is outside, the worst point on the sphere in that cap is one of the vertices.
            # The distance from that vertex to the nearest point in the config is 0 (itself).
            # Wait, the covering radius is the max over y of dist(y, nearest).
            # If the circumcenter is outside the triangle, the worst point in that spherical triangle is a vertex.
            # The distance from a vertex to the nearest point in the config is 0.
            # So the covering radius contribution from this facet is determined by other facets?
            # No, the "cap" is the region where this facet's circumcenter is the closest? No.
            # The covering radius is the max over all y.
            # The points y that are furthest from the nearest configuration point are the circumcenters of the hull facets (if inside) or the vertices (if the circumcenter is outside, but then the vertex is a configuration point, so dist=0).
            # Wait, if the circumcenter is outside the triangle, it means the triangle is "obtuse" on the sphere.
            # The farthest point from the 3 vertices in that spherical triangle is then one of the vertices?
            # No, the farthest point from the set {v1, v2, v3} within the spherical triangle is either the circumcenter (if inside) or the farthest vertex (if outside).
            # But the distance from a vertex to itself is 0.
            # So if the circumcenter is outside, the maximum distance to the nearest of {v1, v2, v3} within the triangle is achieved at a point on the edge?
            # Actually, the function dist(y, {v1, v2, v3}) is convex on the sphere?
            # If the circumcenter is outside, the maximum of the minimum distance to the vertices within the triangle is achieved at the vertex opposite the largest angle?
            # No, let's just use the circumcenter radius as an upper bound, and if it's outside, the true worst point is a vertex, which has distance 0 to itself.
            # But wait, the covering radius is the max over ALL y.
            # If the circumcenter is outside, the worst point in that facet's region is NOT the circumcenter.
            # The worst point is the one that is furthest from the nearest of the 3 vertices.
            # This is a known result: for a spherical triangle, the point furthest from the vertices is either the circumcenter (if inside) or the vertex with the largest angle (if outside)?
            # No, if the circumcenter is outside, the triangle is "obtuse", and the farthest point is the vertex opposite the obtuse angle?
            # Let's just compute the max angle from the circumcenter to the vertices. If they are not equal, the circumcenter is not the center of the minimal covering circle of the 3 points.
            # In that case, the minimal covering circle of the 3 points is determined by 2 of them (the diameter of the pair with the largest distance).
            # So the radius is half the distance between the furthest pair.
            
            # Let's compute the pairwise distances.
            d12 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            d23 = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
            d13 = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
            max_pair_dist = max(d12, d23, d13)
            radius = max_pair_dist / 2.0
            
            # The center of this covering circle is the midpoint of the furthest pair.
            # But for the objective, we just need the radius.
            pass
        else:
            # Circumcenter is inside. Radius is as computed.
            pass
        
        results.append((radius, c, simplex))
    
    return results

def _covering_radius_deg(config):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback: dense sampling
        grid = _fib_sample(30000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    facets = _facet_data(config)
    if not facets:
        return np.degrees(np.pi)
    
    max_radius = 0.0
    for radius, c, simplex in facets:
        if radius > max_radius:
            max_radius = radius
    
    return np.degrees(max_radius)

def _soft_max_objective(params, deadline):
    """
    Smooth surrogate objective: soft-max of facet circumradii.
    """
    if time.time() > deadline - 10:
        return 1e10
    
    pts = params.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    pts = pts / norms
    
    facets = _facet_data(pts)
    if not facets:
        return 1e10
    
    radii = np.array([r for r, _, _ in facets])
    if len(radii) == 0:
        return 1e10
    
    tau = 0.02
    scaled = radii / tau
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return tau * log_sum_exp

def _lloyd_step(config, grid):
    """One step of Lloyd's algorithm using a fixed grid."""
    dots = grid @ config.T
    max_idx = np.argmax(dots, axis=1)
    
    new_config = np.zeros_like(config)
    for i in range(N):
        mask = (max_idx == i)
        count = np.sum(mask)
        if count > 0:
            center = np.mean(grid[mask], axis=0)
            new_config[i] = _normalize(center)
        else:
            new_config[i] = _normalize(np.random.randn(3))
    return new_config

def _refine_worst_facet(config, step_size, deadline, iterations=100):
    """
    Iteratively move the vertices of the worst facet towards the worst center.
    """
    config = config.copy()
    prev_radius = np.pi
    
    for _ in range(iterations):
        if time.time() > deadline - 5:
            break
        
        facets = _facet_data(config)
        if not facets:
            break
        
        max_radius = 0.0
        worst_center = None
        worst_simplex = None
        for radius, c, simplex in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = c
                worst_simplex = simplex
        
        if worst_center is None:
            break
        
        # If radius hasn't improved, reduce step size
        if max_radius >= prev_radius:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        else:
            prev_radius = max_radius
        
        # Move vertices of the worst facet towards worst_center in tangent space
        for idx in worst_simplex:
            v = config[idx]
            dir_vec = worst_center - v
            radial_comp = np.dot(dir_vec, v)
            dir_vec = dir_vec - radial_comp * v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
                config[idx] = v + step_size * dir_vec
                config[idx] = _normalize(config[idx])
    
    return config

def _optimize_from_start(config, deadline):
    """Full optimization pipeline from a starting configuration."""
    start_time = time.time()
    
    # Phase 1: Lloyd's algorithm with a grid
    grid = _fib_sample(10000)
    
    for _ in range(50):
        if time.time() > deadline - 80:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-9):
            break
        config = new_config
    
    # Phase 2: L-BFGS-B optimization on the smooth surrogate objective
    if time.time() < deadline - 30:
        try:
            x0 = config.flatten()
            result = minimize(
                _soft_max_objective,
                x0,
                args=(deadline,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < _soft_max_objective(x0, deadline):
                lbfgs_pts = np.array([_normalize(p) for p in result.x.reshape(N, 3)])
                lbfgs_radius = _covering_radius_deg(lbfgs_pts)
                current_radius = _covering_radius_deg(config)
                if lbfgs_radius < current_radius:
                    config = lbfgs_pts
        except Exception:
            pass
    
    # Phase 3: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for step in step_sizes:
        if time.time() > deadline - 15:
            break
        
        current = _refine_worst_facet(config, step, deadline, iterations=100)
        radius = _covering_radius_deg(current)
        current_radius = _covering_radius_deg(config)
        
        if radius < current_radius:
            config = current
        else:
            config = config  # keep previous
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 seconds safety margin
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Optimize
    config = _optimize_from_start(config, deadline)
    
    # Final normalization and validation
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
    
    return config
