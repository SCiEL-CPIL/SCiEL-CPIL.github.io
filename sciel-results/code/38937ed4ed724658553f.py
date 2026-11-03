"""
Optimized sphere covering with 130 points via Direct Facet-Circumradius Minimization.
Strategy:
1. Use ConvexHull to identify facets (triangles) on the sphere.
2. The covering radius is the maximum circumradius of these facets.
3. Use a smooth-max (softmax) approximation of the maximum circumradius to get a differentiable objective.
4. Compute gradients of this objective with respect to the points on the sphere.
5. Perform gradient descent (projecting back to sphere) to minimize the worst-case coverage.
6. Multi-restart with diverse initializations (Fibonacci, Random, etc.).
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
DEADLINE = time.time() + 145  # 5s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_facet_data(pts):
    """
    Compute normals and circumradii (in radians) for all hull facets.
    Returns:
    normals: (M, 3) array of outward normals
    radii: (M,) array of circumradii in radians
    indices: (M, 3) array of indices of vertices forming each facet
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    normals = hull.equations[:, :3]
    # Ensure normals point outward (away from origin)
    # For a convex hull containing the origin (which it should for sphere covering),
    # the equation is n.x + d = 0. Since origin is inside, n.0 + d < 0 => d < 0.
    # So n points outward if we check n . centroid > 0? 
    # Actually, scipy ConvexHull equations are normalized such that the interior is < 0.
    # So n points OUT of the hull? No, the gradient of the equation points to the positive side.
    # If interior is negative, n points to the exterior.
    # Let's verify: dot(normal, vertex) should be positive for outward normal on unit sphere.
    
    simplices = hull.simplices
    vertices = pts[simplices]
    
    # Calculate circumradius for each triangle
    # Circumradius R of triangle with side lengths a,b,c is abc / (4 * Area)
    # Alternatively, using vectors a,b,c from one vertex:
    # R = |a||b||c| / (2 * |a x b|) ... wait, that's for the triangle in 3D.
    # The circumcenter of the triangle on the sphere is the direction of the normal.
    # The distance from the normal to the vertices is the circumradius.
    # cos(R) = dot(normal, vertex)
    
    # We need the correct outward normal.
    # The normal from scipy is n.
    # Let's check dot(n, vertices). If negative, flip n.
    
    radii = np.zeros(len(normals))
    valid_mask = np.ones(len(normals), dtype=bool)
    
    for i in range(len(normals)):
        n = normals[i]
        # Ensure n points outward
        # The centroid of the facet is a good check
        centroid = vertices[i].mean(axis=0)
        if np.dot(n, centroid) < 0:
            n = -n
            normals[i] = n
            
        # The circumradius is the angle between n and any vertex of the facet
        # Since it's a triangle, all 3 vertices should have the same angle to n
        # (if the circumcenter is inside the triangle).
        # If the circumcenter is outside, the "facet" of the Voronoi cell is not just this triangle?
        # Actually, for sphere covering, the worst case is the circumcenter of the triangle.
        # The angular distance is arccos(dot(n, v)).
        
        dots = np.dot(vertices[i], n)
        # Clip to [-1, 1]
        dots = np.clip(dots, -1.0, 1.0)
        # The circumradius is the max of these? No, they should be equal for the circumcenter.
        # Due to numerical errors, they might differ slightly.
        # The "circumradius" of the triangle is the distance from the circumcenter to the vertices.
        # If the triangle is obtuse, the circumcenter is outside, and the max distance to the 
        # "center" of the cell might be at a vertex or edge midpoint.
        # However, the standard definition of covering radius for a set of points on a sphere
        # is the max distance to the nearest point.
        # The Voronoi diagram on the sphere is formed by the great circles.
        # The vertices of the Voronoi diagram are the circumcenters of the triangles in the Delaunay triangulation.
        # The ConvexHull gives the Delaunay triangulation (if we take the faces).
        # So the circumcenter is indeed along the normal.
        # The angular distance is arccos(dot(n, v)).
        
        # It is possible that the circumcenter is not the "worst point" if the triangle is very large/obtuse?
        # No, the furthest point from the 3 vertices in the spherical triangle is the circumcenter.
        
        r = np.arccos(dots[0])
        radii[i] = r
        # Check if this is a valid facet (normal not degenerate)
        if not np.isfinite(r):
            valid_mask[i] = False
            
    return normals[valid_mask], radii[valid_mask], simplices[valid_mask]

def _smooth_max_grad(values, temperature=0.05):
    """
    Compute the gradient of the smooth max (softmax) of values.
    Returns the weights that sum to 1.
    """
    # Shift for numerical stability
    max_val = np.max(values)
    exp_vals = np.exp((values - max_val) / temperature)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    return weights

def _optimize_facet(pts, max_time, rng, temperature=0.05):
    """
    Optimize points by minimizing the smooth-max of facet circumradii.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_radius = np.inf
    
    step_size = 0.05
    momentum = np.zeros_like(pts)
    mu = 0.9
    
    for step in range(10000):
        if time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE - 5:
            break
            
        normals, radii, simplices = _compute_facet_data(pts)
        
        if normals is None or len(normals) == 0:
            # Fallback: random perturbation
            pts = _normalize(pts + rng.randn(N, 3) * 0.01)
            continue
            
        # Compute smooth max weights
        weights = _smooth_max_grad(radii, temperature)
        
        # The objective is J = sum w_i * r_i
        # We want to minimize J.
        # Gradient of r_i with respect to p_k:
        # r_i = arccos( dot(n_i, p_{v}) )
        # dr_i/dp_v = - n_i / sqrt(1 - dot(n_i, p_v)^2)
        # Note: n_i depends on p. This is complex.
        # Approximation: Assume n_i is fixed for the gradient step (linearization).
        # This is a common approximation in these geometric optimizations.
        
        grad = np.zeros_like(pts)
        
        for i in range(len(normals)):
            w = weights[i]
            if w < 1e-6:
                continue
                
            n = normals[i]
            r = radii[i]
            # sin(r) = sqrt(1 - cos^2(r)) = sqrt(1 - dot^2)
            # To avoid division by zero if r is close to 0 or pi
            sin_r = np.sin(r)
            if sin_r < 1e-6:
                continue
                
            # Factor for gradient magnitude
            factor = w / sin_r
            
            # The gradient of r_i w.r.t p_j is -n_i * factor if p_j is a vertex of facet i
            # Which vertices are in facet i?
            verts = simplices[i]
            for v_idx in verts:
                grad[v_idx] -= factor * n
                
        # Apply gradient descent with momentum
        momentum = mu * momentum + (1 - mu) * grad
        pts = pts - step_size * momentum
        pts = _normalize(pts)
        
        # Evaluate true radius for tracking
        # We already have radii, so max(radii) is the current radius
        cur_max_r = np.max(radii)
        if cur_max_r < best_radius:
            best_radius = cur_max_r
            best_pts = pts.copy()
            
        # Anneal temperature? Or keep constant?
        # Keeping constant is simpler.
        
    return best_pts

def _covering_radius_final(pts):
    """
    Compute the exact covering radius for the final result.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2
    
    max_r = 0.0
    for simplex in hull.simplices:
        p = pts[simplex]
        a, b, c = p[0], p[1], p[2]
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.dot(x, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def construct():
    """
    Main function.
    Strategy: Multi-restart gradient-based optimization on facet circumradii.
    """
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    candidates = []
    
    time_per_restart = 10.0
    max_restarts = int((DEADLINE - start_time - 10) / time_per_restart)
    max_restarts = max(1, min(max_restarts, 12))
    
    for i in range(max_restarts):
        if time.time() > DEADLINE - 15:
            break
            
        # Generate initial points
        if i % 3 == 0:
            # Fibonacci
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts += rng.randn(N, 3) * 0.05
        elif i % 3 == 1:
            # Random
            pts = rng.randn(N, 3)
        else:
            # Perturbed Fibonacci
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j + rng.randn(N) * 0.5
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts += rng.randn(N, 3) * 0.1
            
        pts = _normalize(pts)
        
        # Optimize
        time_left = DEADLINE - time.time() - 10
        if time_left < 2:
            break
            
        t_alloc = min(time_per_restart, time_left / 2.0)
        
        opt_pts = _optimize_facet(pts, t_alloc, rng)
        
        # Evaluate true covering radius
        cr = _covering_radius_final(opt_pts)
        candidates.append((cr, opt_pts))
        
    if not candidates:
        return _normalize(rng.randn(N, 3))
        
    candidates.sort(key=lambda x: x[0])
    best_cr, best_pts = candidates[0]
    
    return _normalize(best_pts)

if __name__ == "__main__":
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_final(pts)))
