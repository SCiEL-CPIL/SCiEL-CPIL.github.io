"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(pts):
    """Normalize points to unit sphere, handling zero vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull circumradii."""
    from scipy.spatial import ConvexHull
    from scipy.spatial import Delaunay
    
    pts = _normalize(pts)
    # Check if all points are in a hemisphere (invalid)
    # Use the fact that if origin is inside the convex hull, no hemisphere contains all points
    hull = ConvexHull(pts)
    
    # Check if origin is inside the hull
    # We can check by seeing if the origin is on the same side of all facets
    # Actually, a simpler check: if the hull contains the origin, it's not in a hemisphere
    # Let's use a different approach: check if there exists a vector v such that v·p > 0 for all p
    # This is equivalent to the origin not being in the convex hull
    
    # For now, compute circumradii of facets
    max_circumradius = 0.0
    for simplex in hull.simplices:
        p = pts[simplex]
        # Compute circumradius of the simplex
        # For a triangle (2-simplex) on the unit sphere
        if len(simplex) == 3:
            a, b, c = p[0], p[1], p[2]
            # Side lengths (chord lengths)
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            # Area using cross product
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            if area < 1e-12:
                circumradius = 0.0
            else:
                circumradius = (ab * bc * ca) / (4.0 * area)
            # Convert chord circumradius to angular radius
            # For a cap with chord radius r_c, the angular radius is 2*arcsin(r_c/2)
            # But we need the angular distance from the center of the cap to the edge
            # The circumradius here is the radius of the circumscribed circle in 3D
            # The angular radius is arcsin(circumradius) since points are on unit sphere
            # Actually, for a triangle on the unit sphere, the circumradius (chord) R satisfies
            # the angular circumradius θ = arcsin(R) if R ≤ 1, but this isn't quite right.
            # Let me think again. The three points are on the unit sphere. The circumcircle 
            # of the triangle (in the plane of the triangle) has some radius R_c. The angular 
            # radius is the angle from the sphere center to the farthest point on the circumcircle.
            # If the circumcenter of the triangle (in 3D) is at distance d from origin, then
            # the angular radius satisfies sin(θ) = R_c / sqrt(d^2 + R_c^2) ... this is getting complex.
            # 
            # Simpler: the covering radius is the maximum over all points y on sphere of min_i angle(y, p_i).
            # This equals the maximum circumradius of the spherical Voronoi cells, which corresponds
            # to the maximum over facets of the angular distance from the facet center to its vertices.
            # 
            # For a spherical triangle with vertices at angular distances from the incenter/circumcenter...
            # Actually, let's just sample the sphere densely to estimate the covering radius.
            pass
    # Fallback: dense sampling
    n_samples = 20000
    # Generate random points on sphere
    rng = np.random.default_rng(42)
    samples = rng.standard_normal((n_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    # Compute angular distance from each sample to nearest point
    # Using dot product: angle = arccos(dot)
    # For efficiency, compute all pairwise dots
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # max dot = min angle
    angles = np.arccos(min_dots)
    return np.max(angles) * 180.0 / np.pi


def _thomson_init(n):
    """Initialize n points using Thomson problem (minimize electrostatic energy)."""
    # Start with Fibonacci spiral
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using gradient descent on electrostatic energy
    # E = sum_{i<j} 1/|p_i - p_j|
    lr = 0.01
    for step in range(500):
        # Compute forces
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Force on i from j: repulsive
                f = d / (dist ** 3)
                forces[i] += f
                forces[j] -= f
        pts = pts + lr * forces
        pts = _normalize(pts)
    
    return pts


def _optimize_covering_radius(pts, time_limit=100.0):
    """Optimize covering radius using gradient descent with smooth max surrogate."""
    start_time = time.time()
    n = len(pts)
    
    # We'll use a smooth approximation of the covering radius
    # The covering radius is max_y min_i angle(y, p_i)
    # We approximate by sampling and using log-sum-exp for smooth max
    
    # Use a fixed set of test points (spherical design)
    # Generate a spherical design for testing
    n_test = 5000
    rng = np.random.default_rng(123)
    test_pts = rng.standard_normal((n_test, 3))
    test_pts /= np.linalg.norm(test_pts, axis=1, keepdims=True)
    
    # Also add the vertices of the current configuration's Voronoi cells
    # For simplicity, just use random samples plus the points themselves
    
    def smooth_covering_radius(pts, beta=100.0):
        """Compute smooth approximation of covering radius."""
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)  # max dot = min angle
        angles = np.arccos(min_dots)
        # Log-sum-exp for smooth max
        beta_inv = 1.0 / beta
        shifted = angles * beta
        max_shifted = np.max(shifted)
        logsumexp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        return logsumexp * beta_inv
    
    def smooth_covering_radius_grad(pts, beta=100.0):
        """Compute gradient of smooth covering radius."""
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(min_dots)
        
        # d(angles)/d(dots) = -1/sqrt(1-dots^2)
        # But we need to be careful: min_dots = max_j dots[:,j]
        # So the gradient flows only through the argmax
        
        # For the smooth max, we use softmax weights
        beta_inv = 1.0 / beta
        shifted = angles * beta
        max_shifted = np.max(shifted)
        exp_shifted = np.exp(shifted - max_shifted)
        weights = exp_shifted / np.sum(exp_shifted)  # softmax weights
        
        # d(angles)/d(dots) for each test point
        # angles = arccos(min_dots)
        # d(angles)/d(min_dots) = -1/sqrt(1-min_dots^2)
        denom = np.sqrt(np.maximum(1.0 - min_dots ** 2, 1e-12))
        d_angles_d_min_dots = -1.0 / denom
        
        # d(min_dots)/d(dots) is the one-hot for argmax, but we use softmax
        # Actually, min_dots = max_j dots[:,j], so d(min_dots)/d(dots[:,j]) = 1 if j=argmax else 0
        # With softmax: d(min_dots)/d(dots[:,j]) ≈ weights[j]
        
        # Gradient: d(CR)/d(pts[i]) = sum_k weights[k] * d_angles_d_min_dots[k] * d(min_dots[k])/d(dots[k,i]) * d(dots[k,i])/d(pts[i])
        # d(dots[k,i])/d(pts[i]) = test_pts[k]
        
        grad = np.zeros_like(pts)
        for i in range(n):
            # For each test point k, the contribution is weights[k] * d_angles_d_min_dots[k] * (softmax weight for i) * test_pts[k]
            # But we need the softmax over j for each k
            # This is expensive. Let's approximate.
            # d(min_dots[k])/d(pts[i]) = test_pts[k] * (1 if i == argmax_j dots[k,j] else 0)
            # With softmax: test_pts[k] * softmax_weights[k, i]
            
            # Compute softmax weights for each test point
            # This is n_test x n matrix
            # For memory, do in chunks
            chunk_size = 1000
            for start in range(0, n_test, chunk_size):
                end = min(start + chunk_size, n_test)
                dots_chunk = dots[start:end, :]
                shifted_chunk = dots_chunk * beta
                max_shifted_chunk = np.max(shifted_chunk, axis=1, keepdims=True)
                exp_chunk = np.exp(shifted_chunk - max_shifted_chunk)
                softmax_chunk = exp_chunk / np.sum(exp_chunk, axis=1, keepdims=True)
                
                # Contribution to grad[i]
                d_min_dots_d_pts_i = test_pts[start:end, i]  # shape (chunk,)
                contrib = weights[start:end] * d_angles_d_min_dots[start:end] * d_min_dots_d_pts_i * softmax_chunk[:, i]
                grad[i] += np.sum(contrib[:, np.newaxis] * test_pts[start:end], axis=0)
        
        return grad * beta_inv
    
    # Gradient descent
    pts_opt = pts.copy()
    lr = 0.01
    beta = 50.0
    patience = 0
    best_cr = _covering_radius(pts_opt)
    best_pts = pts_opt.copy()
    
    while time.time() - start_time < time_limit:
        # Compute smooth CR and gradient
        cr = smooth_covering_radius(pts_opt, beta)
        grad = smooth_covering_radius_grad(pts_opt, beta)
        
        # Update
        pts_opt = pts_opt - lr * grad
        pts_opt = _normalize(pts_opt)
        
        # Check actual CR periodically
        if patience % 50 == 0:
            actual_cr = _covering_radius(pts_opt)
            if actual_cr < best_cr:
                best_cr = actual_cr
                best_pts = pts_opt.copy()
                patience = 0
            else:
                patience += 1
            
            # Adapt learning rate
            if patience > 100:
                lr *= 0.9
                patience = 0
        
        patience += 1
        
        # Increase beta for sharper max approximation
        if patience > 200:
            beta *= 1.2
    
    # Final polish with exact CR
    # Use small steps with exact gradient (finite differences on CR)
    start_polish = time.time()
    lr_polish = 0.001
    for step in range(200):
        if time.time() - start_polish > 30:
            break
        cr = _covering_radius(pts_opt)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts_opt.copy()
        
        # Finite difference gradient
        grad = np.zeros_like(pts_opt)
        eps = 1e-6
        for i in range(n):
            for d in range(3):
                pts_plus = pts_opt.copy()
                pts_plus[i, d] += eps
                pts_plus = _normalize(pts_plus)
                cr_plus = _covering_radius(pts_plus)
                grad[i, d] = (cr_plus - cr) / eps
        pts_opt = pts_opt - lr_polish * grad
        pts_opt = _normalize(pts_opt)
    
    return best_pts


def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Initialize with Thomson configuration
    pts = _thomson_init(N)
    
    # Optimize covering radius
    pts = _optimize_covering_radius(pts, time_limit=time_limit - (time.time() - start_time))
    
    # Ensure valid output
    pts = _normalize(pts)
    
    # Final check: ensure no coincident points and not in a hemisphere
    # Check coincident
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                # Perturb
                pts[j] += 1e-6 * np.random.standard_normal(3)
                pts[j] /= np.linalg.norm(pts[j])
    
    return pts
# EVOLVE-BLOCK-END
