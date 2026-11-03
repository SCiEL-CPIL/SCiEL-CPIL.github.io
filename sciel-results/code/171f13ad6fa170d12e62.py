"""
Optimized sphere covering with 130 points.
Combines fast spectral/electrostatic initialization with precise convex-hull facet refinement.
Strategy:
1. Generate diverse initial configurations (Fibonacci, random, electrostatic).
2. Fast force-directed optimization to reach a good basin.
3. Precise local refinement using convex hull facet circumradii.
   - Uses analytic gradients of the soft-max surrogate for the covering radius.
   - Moves points towards facet circumcenters to reduce facet radii.
4. Final polish with high-precision steps.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 145.0  # 5s safety margin before 150s limit

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        valid: (M,) boolean array of valid facets
    """
    M = len(simplices)
    if M == 0:
        return np.array([]), np.array([]), np.array([])
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        # Covering radius is the angular distance to the circumcenter.
        # If the circumcenter is on the opposite side (dot < 0), the distance is > pi/2.
        # But for a convex hull of points on the sphere, the circumcenter of a facet
        # (the point on the sphere farthest from the facet triangle) is what we want.
        # The circumcenter computed here is the center of the circle passing through a,b,c.
        # There are two such points on the sphere (antipodal). We picked the one with positive dot with centroid.
        # The distance from a to this center is arccos(dot(a, center)).
        # This is the angular radius of the cap.
        radii[valid] = dist
    
    return radii, centers, valid

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    radii, centers, valid = _facet_circumradii(pts, hull.simplices)
    if not np.any(valid):
        return np.pi / 2
    return float(np.max(radii[valid]))

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _spectral_force_step(pts, lr=0.05):
    """
    Vectorized force-directed step using Coulomb repulsion.
    """
    pts = _normalize(pts.copy())
    
    # Compute 3D distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist3_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist3_sq, 1.0) # Avoid div by 0
    dist3 = np.sqrt(dist3_sq)
    
    # Coulomb force: F_i = sum_j (pts[i] - pts[j]) / dist3^3
    inv_dist3 = 1.0 / (dist3 ** 3)
    inv_dist3[dist3 < 1e-6] = 0.0
    F_coulomb = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
    
    # Add a small "spread" force to encourage uniformity on the sphere surface
    # F_spread = -lambda * pts[i] (if centroid is not 0)
    centroid = np.mean(pts, axis=0)
    F_spread = -0.1 * centroid
    
    # Total force
    F = F_coulomb + F_spread
    
    # Normalize force magnitude to prevent large steps
    F_norms = np.linalg.norm(F, axis=1, keepdims=True)
    F_norms[F_norms < 1e-12] = 1.0
    F = F / F_norms
    
    # Update
    new_pts = pts + lr * F
    new_pts = _normalize(new_pts)
    
    return new_pts

def _soft_max_objective_and_grad(points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    M = len(radii)
    
    if M == 0 or not np.any(valid):
        return 0.0, np.zeros_like(points)
    
    # Use only valid facets
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros_like(points)
    
    r_max = np.max(r_valid)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    shifted = beta * (r_valid - r_max)
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(points)
    
    for k in range(len(f_valid)):
        if weights[k] < 1e-10:
            continue
        
        tri = f_valid[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        for idx, p in zip(tri, [p1, p2, p3]):
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                # Gradient of r = arccos(c . p) w.r.t. p is -(c - (c.p)p) / sin(r)
                # We want to decrease r, so move p in direction of -grad = (c - (c.p)p) / sin(r)
                # But we are doing gradient descent on the objective, so we subtract grad.
                # grad_obj = sum w_k * grad_r_k
                # grad_r_k = - (c - (c.p_k)p_k) / sin(r_k)
                # So -grad_obj = sum w_k * (c - (c.p_k)p_k) / sin(r_k)
                # We want to move points to decrease objective, so step = -lr * grad_obj
                # Which is equivalent to step = lr * sum w_k * (c - (c.p_k)p_k) / sin(r_k)
                
                grad_p = -(c - dot_cp * p) / sin_r
                # Project to tangent plane
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += weights[k] * grad_p
    
    return float(log_sum_exp), grad

def _local_refine(pts, max_iters=100, lr_init=0.02, beta=50.0):
    """
    Fine-tune using the actual covering radius gradient (approximated).
    Uses smooth surrogate: soft-max over facet circumradii to get gradients for ALL facets.
    Moves points of large facets towards their circumcenters with weighted forces.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    lr = lr_init
    
    for it in range(max_iters):
        if not _time_ok(2.0):
            break
            
        val, grad = _soft_max_objective_and_grad(pts, beta=beta)
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm
        
        new_pts = pts - lr * grad
        new_pts = _normalize(new_pts)
        
        cr = _covering_radius(new_pts)
        if cr < best_r:
            best_r = cr
            best_pts = new_pts.copy()
            pts = new_pts
            lr *= 0.95 # Decay lr slightly on success
        else:
            lr *= 0.7
            if lr < 0.0005:
                break
        
        if lr < 0.0005:
            break
            
    return best_pts, best_r

def construct():
    """
    1. Generate diverse initial configs.
    2. Run fast spectral force-directed optimization (no hull).
    3. Refine the best candidate with slow hull-based local search.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 111, 222, 333]:
        inits.append(_fibonacci_spiral(N, jitter=0.05, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    
    # Sort by initial covering radius (fast check)
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])
    
    # Phase 1: Fast Spectral Optimization
    for r0, p0 in init_scores:
        if not _time_ok(15.0):
            break
            
        pts = p0.copy()
        # Run many fast steps
        for it in range(1000):
            if not _time_ok(10.0):
                break
            pts = _spectral_force_step(pts, lr=0.01)
            
        # Evaluate
        cr = _covering_radius(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
            
    # Phase 2: Slow Refinement of the Best
    if best_pts is not None and _time_ok(10.0):
        best_pts, best_r = _local_refine(best_pts, max_iters=100, lr_init=0.015, beta=50.0)
        
    # Phase 3: High-precision polish pass
    if best_pts is not None and _time_ok(5.0):
        # Final fine-tuning with very small steps and multiple restarts
        polish_lr = 0.003
        for polish_iter in range(50):
            if not _time_ok(2.0):
                break
            
            try:
                hull = ConvexHull(best_pts)
            except Exception:
                break
                
            # Find top 5 largest facets
            facet_radii = []
            radii, centers, valid = _facet_circumradii(best_pts, hull.simplices)
            for idx in range(len(radii)):
                if valid[idx]:
                    facet_radii.append((radii[idx], idx, centers[idx]))
            
            if not facet_radii:
                break
                
            facet_radii.sort(key=lambda x: -x[0])
            top_facets = facet_radii[:5]
            
            # Move points of top facets towards their circumcenters
            for r, idx, cc in top_facets:
                simplex = hull.simplices[idx]
                for i in simplex:
                    best_pts[i] = best_pts[i] + polish_lr * (cc - best_pts[i])
            
            best_pts = _normalize(best_pts)
            cr = _covering_radius(best_pts)
            if cr < best_r:
                best_r = cr
            else:
                polish_lr *= 0.8
                if polish_lr < 0.0005:
                    break
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius(pts)))
    print("Time: %.1f s" % (time.time() - t0))
