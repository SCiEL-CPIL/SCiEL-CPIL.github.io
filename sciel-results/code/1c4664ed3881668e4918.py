"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: grid
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of spherical triangle a,b,c
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(u, n) > 0:
            circum_center = u
        else:
            circum_center = -u
            
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
        
    return max_cr

def _covering_radius_grid(pts, n_grid=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_lloyd(pts, max_iter=1000, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        incident_facets = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            centroid = np.sum(incident_facets[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
                
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
            
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _optimize_facet_shrink(pts, max_iter=1000, initial_step=0.5, decay=0.995, beta_start=2.0, beta_end=20.0):
    """
    Optimizes the configuration by moving each point towards the weighted
    average of the circumcenters of its incident facets.
    Uses adaptive step size and increasing beta to focus on worst facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        
        max_cr_global = 0.0
        facet_data = []

        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
                
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, cc, cr))
            
        if max_cr_global < 1e-6:
            break

        # Interpolate beta from beta_start to beta_end
        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        
        for simplex, cc, cr in facet_data:
            w = math.exp(beta * (cr - max_cr_global))
            
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
                
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        # Decay step size
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts

def _smooth_covering_radius(pts, beta=500.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    We sample points on the sphere and compute the soft-max of distances to nearest point.
    """
    pts = _normalize(pts)
    # Use a fixed set of sample points for consistency
    # For speed, we use a moderate number of samples
    n_samples = 1000
    # Pre-computed sample points (uniform on sphere)
    # We generate them once here, but in a real optimization loop, this should be cached
    # For now, we generate them with a fixed seed
    rng = np.random.RandomState(42)
    samples = rng.randn(n_samples, 3)
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute distances from samples to all pts
    # dots[i, j] = dot(sample_i, pt_j)
    dots = samples @ pts.T
    dots = np.clip(dots, -1, 1)
    
    # For each sample, the distance to the nearest pt is arccos(max_j dots[i, j])
    # We want to minimize the max over samples of this distance.
    # Let d_i = arccos(max_j dots[i, j]).
    # We approximate max(d_i) by log-sum-exp(beta * d_i) / beta.
    
    # But arccos is non-linear.
    # Alternative: minimize max_j (distance).
    # Let's just compute the actual max distance on the grid for the objective,
    # but that's not smooth.
    
    # Let's stick to the grid-based max distance but make it smooth?
    # No, the directive suggests using log-sum-exp on the circumradii of facets.
    # Let's compute the circumradii of the convex hull facets and use log-sum-exp on them.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 10.0  # Large penalty for invalid hull
        
    crs = []
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
        else:
            u = v / norm_v
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            if np.dot(n, a) < 0:
                n = -n
            if np.dot(u, n) > 0:
                cc = u
            else:
                cc = -u
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
        crs.append(cr)
        
    if not crs:
        return 10.0
        
    crs = np.array(crs)
    # Log-sum-exp
    m = np.max(crs)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(crs - m))) / beta

def _optimize_lbfgs(pts, max_iter=100, time_limit=10.0):
    """
    Optimize points using L-BFGS-B on the sphere.
    We parameterize points by their spherical coordinates (theta, phi) or use
    a retraction method.
    Here we use a simple approach: optimize the 3D coordinates with a penalty
    for not being on the sphere, or use a tangent space retraction.
    
    For simplicity and robustness, we will use a "tangent space" step:
    1. Compute gradient of smooth objective w.r.t. pts (via finite diff or analytic).
    2. Project gradient to tangent space.
    3. Take a step in tangent space.
    4. Retract to sphere.
    
    Since we don't have an easy analytic gradient for the log-sum-exp of circumradii,
    we will use finite differences for the gradient.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    def objective(x):
        # x is flattened pts
        p = x.reshape(n, 3)
        p = _normalize(p)
        return _smooth_covering_radius(p, beta=500.0)
    
    # We will use scipy.optimize.minimize with method L-BFGS-B
    # But we need to handle the sphere constraint.
    # L-BFGS-B doesn't handle non-linear constraints well.
    # Instead, we will do a manual gradient descent with retraction.
    
    start_time = time.time()
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient via finite differences
        grad = np.zeros_like(pts)
        eps = 1e-4
        f0 = objective(pts.ravel())
        
        for i in range(n):
            for d in range(3):
                pts_p = pts.copy()
                pts_p[i, d] += eps
                pts_p[i] = pts_p[i] / np.linalg.norm(pts_p[i])
                fp = objective(pts_p.ravel())
                
                pts_m = pts.copy()
                pts_m[i, d] -= eps
                pts_m[i] = pts_m[i] / np.linalg.norm(pts_m[i])
                fm = objective(pts_m.ravel())
                
                grad[i, d] = (fp - fm) / (2 * eps)
                
        # Project gradient to tangent space
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
            
        # Step
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Adaptive learning rate
        # If objective increases, reduce lr
        f1 = objective(pts.ravel())
        if f1 > f0:
            lr *= 0.5
        else:
            lr = min(lr * 1.05, 0.1)
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        if restart_count == 1:
            pts = _fibonacci_points(N)
        else:
            # Perturb best solution
            noise_scale = 0.1 * (best_cr / 0.3)
            if noise_scale > 0.5:
                noise_scale = 0.5
            if noise_scale < 0.01:
                noise_scale = 0.01
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # 1. Lloyd's relaxation
        pts = _spherical_lloyd(pts, max_iter=50)
        
        # 2. Facet shrink
        pts = _optimize_facet_shrink(pts, max_iter=100, initial_step=0.3, decay=0.95)
        
        # 3. New: L-BFGS-style polish with smooth surrogate
        if time.time() < deadline - 15:
            pts = _optimize_lbfgs(pts, max_iter=50, time_limit=10.0)
            # Final facet shrink to clean up
            pts = _optimize_facet_shrink(pts, max_iter=50, initial_step=0.1, decay=0.9)
            
        if time.time() > deadline:
            break
            
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
