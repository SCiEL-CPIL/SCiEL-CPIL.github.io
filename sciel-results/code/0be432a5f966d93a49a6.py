"""
Sphere covering with 130 points - optimized covering radius.
Uses a combination of:
1. Fibonacci sphere initialization
2. Log-sum-exp smooth approximation for gradient-based optimization
3. Convex hull circumradius evaluation
4. Local refinement of worst facets
"""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_exact(points):
    """Compute exact covering radius using convex hull circumcircles."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_r = 0.0
        for simplex in facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            if n_pts == 3:
                a, b, c = pts
                ab = b - a
                ac = c - a
                bc = c - b
                norm_ab = np.linalg.norm(ab)
                norm_ac = np.linalg.norm(ac)
                norm_bc = np.linalg.norm(bc)
                # Area of triangle
                cross = np.cross(ab, ac)
                area = 0.5 * np.linalg.norm(cross)
                if area < 1e-15:
                    continue
                # Circumradius = (abc) / (4 * area) where a,b,c are side lengths
                R = (norm_ab * norm_ac * norm_bc) / (4.0 * 2.0 * area)
                # Convert to angular distance
                # For points on unit sphere, circumradius R relates to angular radius theta by:
                # R = 2*sin(theta/2) for equilateral, but generally we need the cap
                # Actually, for a triangle on the unit sphere, the circumcenter is the
                # point on the sphere equidistant from all three vertices.
                # Let's compute it directly.
                # Normal to the plane
                n = np.cross(ab, ac)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-15:
                    continue
                n = n / n_norm
                # Check if circumcenter is on the "outside" (positive side)
                # The circumcenter in 3D: find point p on line a + t*n equidistant from a,b,c
                # |p-a|^2 = |p-b|^2 => 2*(p-a)·(b-a) = |b-a|^2
                # p = a + t*n, so 2*(t*n - (a-a))·(b-a) = |b-a|^2 => 2*t*(n·(b-a)) = |b-a|^2
                # t = |b-a|^2 / (2 * n·(b-a))
                # But we need the point ON THE SPHERE that is the cap center.
                # The cap center is the normalized vector from origin to the circumcenter of the spherical triangle.
                # For a spherical triangle, the circumcenter (pole of the great circle through the 3 points)
                # is just the normalized cross product... no, that's the normal to the plane.
                # The angular distance from the pole to each vertex is the same only if the triangle is
                # equiangular. In general, the covering radius is determined by the facet whose
                # circumcircle has the largest angular radius.
                # For a planar triangle inscribed in the unit sphere, the circumradius of the circle
                # (in 3D) that passes through the three points and has its center on the line from
                # origin through the triangle's circumcenter gives us the angular radius.
                # Let me just compute: the three points define a circle. The center of this circle
                # is the intersection of the plane with the line from origin perpendicular to the plane.
                # Distance from origin to plane: d = |a·n|
                # Radius of the circle: r_circ = sqrt(1 - d^2) (since sphere radius is 1)
                # Angular radius: theta = arccos(d) = arcsin(r_circ)
                d = abs(np.dot(a, n))
                d = min(d, 1.0)
                theta = np.arccos(d)
                max_r = max(max_r, theta)
            else:
                # For non-triangular facets, use centroid approximation
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_theta = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    theta = np.arccos(cos_theta)
                    max_r = max(max_r, theta)
        
        return np.degrees(max_r)
    except Exception:
        return np.inf

def _covering_radius_fast(points):
    """Fast vectorized covering radius computation."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        tri_facets = [f for f in facets if len(f) == 3]
        
        max_r = 0.0
        if len(tri_facets) > 0:
            tri_facets = np.array(tri_facets)
            a = points[tri_facets[:, 0]]
            b = points[tri_facets[:, 1]]
            c = points[tri_facets[:, 2]]
            
            ab = b - a
            ac = c - a
            
            n = np.cross(ab, ac)
            n_norms = np.linalg.norm(n, axis=1)
            
            valid = n_norms > 1e-15
            if np.any(valid):
                n = n[valid] / n_norms[valid, np.newaxis]
                a_valid = a[valid]
                
                d = np.abs(np.einsum('ij,ij->i', a_valid, n))
                d = np.clip(d, 0.0, 1.0)
                thetas = np.arccos(d)
                max_r = max(max_r, np.max(thetas))
        
        non_tri = [f for f in facets if len(f) != 3]
        for simplex in non_tri:
            pts = points[simplex]
            centroid = np.mean(pts, axis=0)
            nc = np.linalg.norm(centroid)
            if nc < 1e-15:
                continue
            centroid = centroid / nc
            for p in pts:
                cos_t = np.clip(np.dot(centroid, p), -1.0, 1.0)
                max_r = max(max_r, np.arccos(cos_t))
        
        return np.degrees(max_r)
    except Exception:
        return np.inf

def _logsumexp_covering_radius(points, temperature=0.01):
    """
    Smooth approximation of covering radius using log-sum-exp over all pairs.
    For each point y on sphere (sampled), compute distance to nearest point.
    We approximate by checking against the convex hull facets.
    Actually, let's use a different approach: for each pair of nearby points,
    compute the bisecting cap radius.
    """
    # This is a smoother surrogate: for each convex hull facet, compute its
    # circumradius, then use a smooth max (log-sum-exp) instead of hard max.
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        thetas = []
        for simplex in facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            if n_pts == 3:
                a, b, c = pts
                ab = b - a
                ac = c - a
                n = np.cross(ab, ac)
                nn = np.linalg.norm(n)
                if nn < 1e-15:
                    continue
                n = n / nn
                d = abs(np.dot(a, n))
                d = min(d, 1.0)
                theta = np.arccos(d)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                # Use max distance to vertices as approximation
                max_dist = max(np.arccos(np.clip(np.dot(centroid, p), -1, 1)) for p in pts)
                theta = max_dist
            thetas.append(theta)
        
        if not thetas:
            return np.inf
        
        thetas = np.array(thetas)
        # Log-sum-exp for smooth max
        t = max(temperature, 1e-10)
        shifted = thetas / t
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        smooth_max = t * log_sum
        
        return np.degrees(smooth_max)
    except Exception:
        return np.inf

def _gradient_step(points, temperature=0.01, lr=0.01):
    """
    Compute gradient of the log-sum-exp covering radius w.r.t. point positions.
    For each facet, the circumradius theta depends on the three points.
    d(theta)/d(point_i) can be computed via chain rule.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        grad = np.zeros_like(points)
        t = max(temperature, 1e-10)
        
        # First compute all thetas
        thetas = np.zeros(len(facets))
        facet_data = []
        for i, simplex in enumerate(facets):
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                thetas[i] = -np.inf
                facet_data.append(None)
                continue
            if n_pts == 3:
                a, b, c = pts
                ab = b - a
                ac = c - a
                n = np.cross(ab, ac)
                nn = np.linalg.norm(n)
                if nn < 1e-15:
                    thetas[i] = -np.inf
                    facet_data.append(None)
                    continue
                n = n / nn
                d = abs(np.dot(a, n))
                d = min(d, 1.0)
                theta = np.arccos(d)
                thetas[i] = theta
                facet_data.append((a, b, c, n, d, theta))
            else:
                thetas[i] = -np.inf
                facet_data.append(None)
        
        # Weight each facet by exp(theta_i / t) / sum(exp(theta_j / t))
        shifted = thetas / t
        valid_mask = np.isfinite(shifted)
        if not np.any(valid_mask):
            return points
        
        max_shifted = np.max(shifted[valid_mask])
        weights = np.exp(shifted - max_shifted)
        weights[~valid_mask] = 0.0
        total_weight = np.sum(weights)
        if total_weight < 1e-30:
            return points
        weights /= total_weight
        
        # For each triangle facet, compute gradient of theta w.r.t. a, b, c
        # theta = arccos(d) where d = |a·n| and n = (b-a)×(c-a) / ||(b-a)×(c-a)||
        # This is complex; let's use a simpler gradient estimate via finite differences
        # or just move points away from the worst facet center.
        
        # Alternative: for the weighted worst facets, push their points apart
        for i, fd in enumerate(facet_data):
            if fd is None or weights[i] < 1e-10:
                continue
            a, b, c, n, d, theta = fd
            # The facet center (on sphere) is in direction n (or -n)
            # Push points a, b, c away from each other and from the center
            # Simple heuristic: move each vertex in direction that increases the
            # minimum distance to the other two vertices
            for pt in [a, b, c]:
                # Direction: away from centroid of the other two
                # This is a rough gradient
                pass
        
        # Simpler approach: use the fact that to reduce the circumradius,
        # we want to make the triangle more "equilateral" on the sphere
        # and/or move the circumcenter toward the origin (decrease d).
        # Moving points along the normal direction reduces d.
        for i, fd in enumerate(facet_data):
            if fd is None or weights[i] < 1e-10:
                continue
            a, b, c, n, d, theta = fd
            # Move each point slightly in the direction of -n (toward origin side)
            # This decreases |a·n|
            move = -n * weights[i] * lr
            idx = facets[i]
            for j in range(3):
                grad[idx[j]] += move
        
        # Apply gradient
        new_points = points + grad
        # Renormalize
        norms = np.linalg.norm(new_points, axis=1)
        norms[norms < 1e-10] = 1.0
        new_points = new_points / norms[:, np.newaxis]
        
        return new_points
    except Exception:
        return points

def _optimize_logsumexp(points, max_time, temperature=0.01, lr=0.01):
    """Optimize using log-sum-exp smooth covering radius with gradient steps."""
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_radius = _covering_radius_fast(best)
    current_radius = _logsumexp_covering_radius(current, temperature)
    
    step = 0
    while time.time() - start_time < max_time:
        # Gradient step
        new_points = _gradient_step(current, temperature, lr)
        new_radius = _covering_radius_fast(new_points)
        
        if new_radius < best_radius:
            best = new_points.copy()
            best_radius = new_radius
            current = new_points.copy()
        elif step % 10 == 0:
            # Occasionally try to recover
            current = best.copy()
        
        # Adapt learning rate
        step += 1
        if step % 50 == 0:
            lr *= 0.95
            temperature *= 0.99
    
    return best, best_radius

def _local_refine(points, max_time):
    """Local refinement focusing on worst facets."""
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_radius = _covering_radius_fast(best)
    
    step_size = 0.02
    while time.time() - start_time < max_time:
        # Find worst facet
        try:
            hull = ConvexHull(current)
            facets = hull.simplices
            
            worst_idx = -1
            worst_theta = 0.0
            for i, simplex in enumerate(facets):
                pts = current[simplex]
                if len(pts) < 3:
                    continue
                a, b, c = pts[0], pts[1], pts[2]
                ab = b - a
                ac = c - a
                n = np.cross(ab, ac)
                nn = np.linalg.norm(n)
                if nn < 1e-15:
                    continue
                n = n / nn
                d = abs(np.dot(a, n))
                d = min(d, 1.0)
                theta = np.arccos(d)
                if theta > worst_theta:
                    worst_theta = theta
                    worst_idx = i
        except Exception:
            break
        
        if worst_idx < 0:
            break
        
        simplex = facets[worst_idx]
        improved = False
        
        # Try to improve the worst facet by moving its vertices
        for _ in range(2):
            for j in range(3):
                idx = simplex[j]
                for _ in range(5):
                    perturb = np.random.randn(3) * step_size
                    old = current[idx].copy()
                    new_pt = current[idx] + perturb
                    nn = np.linalg.norm(new_pt)
                    if nn < 1e-10:
                        continue
                    new_pt = new_pt / nn
                    current[idx] = new_pt
                    
                    new_radius = _covering_radius_fast(current)
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best = current.copy()
                        improved = True
                    else:
                        current[idx] = old
        
        if not improved:
            step_size *= 0.85
            if step_size < 1e-7:
                break
        
        current = best.copy()
    
    return best, best_radius

def _isotropic_correction(points):
    """Apply isotropic correction to the point distribution."""
    M = points.T @ points
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-10, None)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    new_points = points @ W.T
    norms = np.linalg.norm(new_points, axis=1)
    norms[norms < 1e-10] = 1.0
    new_points = new_points / norms[:, np.newaxis]
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with Fibonacci sphere
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _covering_radius_fast(best_points)
    
    # Phase 1: Isotropic correction
    if time.time() < deadline - 30.0:
        iso_points = _isotropic_correction(points)
        iso_radius = _covering_radius_fast(iso_points)
        if iso_radius < best_radius:
            best_points = iso_points
            best_radius = iso_radius
            points = iso_points
    
    # Phase 2: Log-sum-exp gradient optimization
    if time.time() < deadline - 20.0:
        lse_points, lse_radius = _optimize_logsumexp(points, max_time=30.0, temperature=0.01, lr=0.02)
        if lse_radius < best_radius:
            best_points = lse_points
            best_radius = lse_radius
            points = lse_points
    
    # Phase 3: Local refinement of worst facets
    if time.time() < deadline - 10.0:
        lr_points, lr_radius = _local_refine(points, max_time=20.0)
        if lr_radius < best_radius:
            best_points = lr_points
            best_radius = lr_radius
            points = lr_points
    
    # Phase 4: Another log-sum-exp pass with smaller temperature
    if time.time() < deadline - 5.0:
        lse2_points, lse2_radius = _optimize_logsumexp(points, max_time=10.0, temperature=0.005, lr=0.01)
        if lse2_radius < best_radius:
            best_points = lse2_points
            best_radius = lse2_radius
            points = lse2_points
    
    # Phase 5: Final local polish
    if time.time() < deadline - 2.0:
        lr2_points, lr2_radius = _local_refine(points, max_time=5.0)
        if lr2_radius < best_radius:
            best_points = lr2_points
            best_radius = lr2_radius
    
    return best_points
# EVOLVE-BLOCK-END
