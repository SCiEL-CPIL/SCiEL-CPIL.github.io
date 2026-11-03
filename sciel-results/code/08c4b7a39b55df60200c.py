"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
        n = n / np.linalg.norm(n)
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

def _jittered_fibonacci(n, seed=42, jitter=0.1):
    """Fibonacci lattice with random jitter to break symmetry."""
    np.random.seed(seed)
    pts = _fibonacci_points(n)
    # Add small random perturbations
    noise = jitter * np.random.randn(n, 3)
    pts = pts + noise
    return _normalize(pts)

def _spherical_lloyd(pts, max_iter=200, tol=1e-9):
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
            
        # Map point index to list of incident facets
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
            
            # Ensure normal points away from origin
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        # Compute new positions
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

def _soft_covering_radius(pts, beta=100.0):
    """
    Compute a smooth approximation of the covering radius.
    Uses softmax over angular distances to create a differentiable objective.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute all pairwise angular distances
    dots = pts @ pts.T
    dots = np.clip(dots, -1, 1)
    dists = np.arccos(dots)
    
    # For each point, find the max distance to any other point
    # The covering radius is half the max distance (for the worst pair)
    # But we need the true covering radius, which is related to the convex hull facets
    
    # Alternative: use a grid-based smooth approximation
    # Sample many points on the sphere and compute soft max distance
    n_samples = 500
    np.random.seed(123)
    # Uniform random points on sphere
    vecs = np.random.randn(n_samples, 3)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    
    # Distance from each sample to nearest config point
    dots = vecs @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    angles = np.arccos(min_dots)
    
    # Soft max approximation
    return np.log(np.sum(np.exp(beta * angles))) / beta

def _gradient_descent(pts, max_iter=500, lr=0.01, beta=500.0):
    """
    Gradient descent on the smooth covering radius objective.
    Uses finite differences to compute gradients.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        # Compute current objective
        obj = _soft_covering_radius(pts, beta=beta)
        
        # Compute gradient via finite differences
        grad = np.zeros_like(pts)
        eps = 1e-5
        for i in range(n):
            for d in range(3):
                pts_p = pts.copy()
                pts_p[i, d] += eps
                pts_p[i] = pts_p[i] / np.linalg.norm(pts_p[i])
                obj_p = _soft_covering_radius(pts_p, beta=beta)
                
                pts_m = pts.copy()
                pts_m[i, d] -= eps
                pts_m[i] = pts_m[i] / np.linalg.norm(pts_m[i])
                obj_m = _soft_covering_radius(pts_m, beta=beta)
                
                grad[i, d] = (obj_p - obj_m) / (2 * eps)
        
        # Project gradient onto tangent space
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Gradient descent step
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Reduce learning rate over time
        lr *= 0.995
        
    return pts

def _facet_shrink_polish(pts, max_iter=300, step_size=0.05):
    """
    Polish by iteratively moving the worst facet's vertices toward its circumcenter.
    """
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
            n_face = n_face / np.linalg.norm(n_face)
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_pts = _jittered_fibonacci(N, seed=42, jitter=0.05)
    best_cr = _covering_radius(best_pts)
    
    # Multi-stage optimization with multiple restarts
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:  # Leave 20s for final evaluation
            break
        
        # Initialize
        if restart == 0:
            pts = _jittered_fibonacci(N, seed=42, jitter=0.05)
        elif restart == 1:
            pts = _fibonacci_points(N)
        elif restart == 2:
            # Random initialization
            vecs = np.random.randn(N, 3)
            pts = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        else:
            # Perturb best solution
            noise_scale = 0.2 / (1 + restart * 0.15)
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
        
        # Stage 1: Lloyd's algorithm for global structure
        pts = _spherical_lloyd(pts, max_iter=150)
        
        # Stage 2: Gradient descent with high beta for fine-tuning
        if time.time() < deadline - 15:
            pts = _gradient_descent(pts, max_iter=200, lr=0.005, beta=500.0)
        
        # Stage 3: Facet shrink polish
        pts = _facet_shrink_polish(pts, max_iter=200, step_size=0.03)
        
        # Evaluate
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
