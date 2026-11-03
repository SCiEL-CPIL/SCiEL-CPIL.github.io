"""Rewritten program for covering the sphere with 50 equal caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
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

def _covering_radius_grid(pts, n_grid=80):
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

def _random_points(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vecs = np.random.randn(n, 3)
    return _normalize(vecs)

def _spherical_lloyd(pts, max_iter=200, tol=1e-9):
    """Spherical Lloyd's algorithm (Voronoi relaxation)."""
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

def _force_relax(pts, target_r, max_iter=300, dt=0.01):
    """
    Force-based relaxation: repel points that are too close, attract to fill gaps.
    target_r is the target covering radius in radians.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        forces = np.zeros_like(pts)
        
        # Compute pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1, 1)
        dists = np.arccos(dots)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    # Very close points: strong repulsion
                    forces[i] += 10.0 * (pts[i] - pts[j])
                    forces[j] += 10.0 * (pts[j] - pts[i])
                    continue
                if d < target_r * 0.5:
                    # Too close: repel
                    strength = (target_r * 0.5 - d) / target_r
                    # Tangential component of repulsion
                    vec = pts[j] - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vt = np.linalg.norm(vec_tan)
                    if norm_vt > 1e-10:
                        forces[i] += strength * (vec_tan / norm_vt)
        
        # Gap-filling: attract points toward directions with large empty caps
        # Sample directions and find worst-covered region
        n_samples = 200
        np.random.seed(42 + it)
        vecs = np.random.randn(n_samples, 3)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        dots = vecs @ pts.T
        dots = np.clip(dots, -1, 1)
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(min_dots)
        
        # Find the worst direction
        worst_idx = np.argmax(angles)
        worst_dir = vecs[worst_idx]
        worst_angle = angles[worst_idx]
        
        if worst_angle > target_r:
            # Attract the nearest point toward the worst direction
            dots_to_pts = pts @ worst_dir
            nearest_idx = np.argmax(dots_to_pts)
            vec = worst_dir - pts[nearest_idx]
            vec_tan = vec - np.dot(vec, pts[nearest_idx]) * pts[nearest_idx]
            norm_vt = np.linalg.norm(vec_tan)
            if norm_vt > 1e-10:
                strength = min(1.0, (worst_angle - target_r) / target_r) * 0.5
                forces[nearest_idx] += strength * (vec_tan / norm_vt)
        
        # Apply forces with adaptive step
        for i in range(n):
            f = forces[i]
            f_tan = f - np.dot(f, pts[i]) * pts[i]
            norm_ft = np.linalg.norm(f_tan)
            if norm_ft > 1e-10:
                step = dt * min(1.0, norm_ft)
                pts[i] += step * (f_tan / norm_ft)
        
        pts = _normalize(pts)
        
        # Check convergence
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-8:
            break
            
    return pts

def _facet_shrink_polish(pts, max_iter=300, step_size=0.05):
    """Polish by iteratively moving the worst facet's vertices toward its circumcenter."""
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        max_cr = 0.0
        max_facet = None
        max_cc = None
        
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
            norm_nf = np.linalg.norm(n_face)
            if norm_nf < 1e-12:
                continue
            n_face = n_face / norm_nf
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
            
            if cr > max_cr:
                max_cr = cr
                max_facet = simplex
                max_cc = cc
                
        if max_facet is None:
            break
            
        for idx in max_facet:
            p = pts[idx]
            vec = max_cc - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-10:
                step = step_size * min(1.0, max_cr)
                pts[idx] = p + step * (vec_tan / norm_vec)
                
        pts = _normalize(pts)
        
        if max_cr < 1e-7:
            break
            
    return pts

def _soft_max_covering_radius(pts, beta=100.0, n_samples=500):
    """Smooth approximation of covering radius using soft max."""
    pts = _normalize(pts)
    np.random.seed(123)
    vecs = np.random.randn(n_samples, 3)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    
    dots = vecs @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    angles = np.arccos(min_dots)
    
    return np.log(np.sum(np.exp(beta * angles))) / beta

def _gradient_descent(pts, max_iter=300, lr=0.01, beta=500.0):
    """Gradient descent on the smooth covering radius objective."""
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        obj = _soft_max_covering_radius(pts, beta=beta)
        
        grad = np.zeros_like(pts)
        eps = 1e-5
        for i in range(n):
            for d in range(3):
                pts_p = pts.copy()
                pts_p[i, d] += eps
                pts_p[i] = pts_p[i] / np.linalg.norm(pts_p[i])
                obj_p = _soft_max_covering_radius(pts_p, beta=beta, n_samples=200)
                
                pts_m = pts.copy()
                pts_m[i, d] -= eps
                pts_m[i] = pts_m[i] / np.linalg.norm(pts_m[i])
                obj_m = _soft_max_covering_radius(pts_m, beta=beta, n_samples=200)
                
                grad[i, d] = (obj_p - obj_m) / (2 * eps)
        
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        lr *= 0.995
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    # Multi-stage optimization with multiple restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25:  # Leave 25s for final evaluation
            break
        
        # Initialize
        if restart == 0:
            pts = _fibonacci_points(N)
        elif restart == 1:
            pts = _random_points(N, seed=100 + restart)
        elif restart == 2:
            # Jittered Fibonacci
            pts = _fibonacci_points(N)
            noise = 0.05 * np.random.randn(N, 3)
            pts = pts + noise
            pts = _normalize(pts)
        elif restart == 3:
            # Perturb best solution with larger noise
            noise = 0.3 * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
        else:
            # Perturb best solution with decreasing noise
            noise_scale = 0.2 / (1 + (restart - 4) * 0.2)
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
        
        # Stage 1: Lloyd's algorithm for global structure
        pts = _spherical_lloyd(pts, max_iter=100)
        
        # Stage 2: Force-based relaxation
        if time.time() < deadline - 20:
            current_cr = _covering_radius(pts)
            target_r = current_cr * 0.9  # Try to reduce by 10%
            pts = _force_relax(pts, target_r, max_iter=200, dt=0.005)
        
        # Stage 3: Facet shrink polish
        if time.time() < deadline - 10:
            pts = _facet_shrink_polish(pts, max_iter=150, step_size=0.03)
        
        # Evaluate
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
