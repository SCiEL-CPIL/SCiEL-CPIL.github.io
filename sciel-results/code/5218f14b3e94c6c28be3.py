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

def _repulsion_energy(pts):
    """
    Compute a smooth repulsion energy that encourages uniform distribution.
    We use the sum of inverse distances to the power p.
    """
    pts = _normalize(pts)
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    # dist^2 = 2 - 2 * dot
    dots = pts @ pts.T
    dots = np.clip(dots, -1, 1)
    dists = np.arccos(dots)
    
    # Set diagonal to 1 to avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Use a power p > 2 to emphasize close pairs (repulsion)
    p = 4
    energy = np.sum(1.0 / (dists ** p))
    return energy

def _repulsion_gradient(pts):
    """
    Compute the gradient of the repulsion energy.
    E = sum_{i!=j} d_{ij}^{-p}
    dE/dx_i = sum_{j!=i} -p * d_{ij}^{-p-1} * (d d_{ij}/d x_i)
    d_{ij} = arccos(x_i . x_j)
    d d_{ij}/d x_i = - (x_j - (x_i . x_j) x_i) / sqrt(1 - (x_i . x_j)^2)
    Note: The gradient on the sphere is the projection of the Euclidean gradient.
    Since we are optimizing on the sphere, we can compute the Euclidean gradient and project.
    """
    pts = _normalize(pts)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    if n < 2:
        return grad
        
    dots = pts @ pts.T
    dots = np.clip(dots, -1, 1)
    dists = np.arccos(dots)
    
    p = 4
    # We need 1/sqrt(1-dot^2) which is 1/sin(d)
    # sin(d) = sqrt(1 - cos^2(d)) = sqrt(1 - dot^2)
    sin_dists = np.sqrt(np.maximum(1.0 - dots**2, 1e-12))
    
    # Vectorized gradient calculation
    # For each pair i, j, the force on i is proportional to - (x_j - dot x_i) / sin(d) * d/dx (d^{-p})
    # Actually, let's use the formula for the gradient of arccos(u) w.r.t x: - (y - (x.y)x) / sqrt(1-(x.y)^2)
    
    # This is O(N^2) which is fine for N=50
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-6:
                continue
            sin_d = sin_dists[i, j]
            # Gradient of d_{ij} w.r.t x_i
            # grad_d = -(x_j - dot(x_i, x_j) * x_i) / sin(d)
            dot_ij = dots[i, j]
            vec = pts[j] - dot_ij * pts[i]
            grad_d = -vec / sin_d
            
            # Gradient of d^{-p} is -p * d^{-p-1} * grad_d
            factor = -p * (d ** (-p - 1))
            grad[i] += factor * grad_d
            
    # Project gradient to the tangent space of the sphere
    # grad_tan = grad - (grad . x) x
    for i in range(n):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
        
    return grad

def _polish_facets(pts, max_iter=200, step=0.1):
    """
    Local polish using facet circumcenters to minimize covering radius.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
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

        # Use a fixed beta for stability in polishing
        beta = 5.0
        
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
                        pts[i] = pts[i] + step * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        # Decay step size
        step *= 0.95
        if step < 1e-8:
            break

    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    # List of initializations
    initializations = []
    initializations.append(_fibonacci_points(N))
    
    # Add random perturbations of fibonacci
    for _ in range(5):
        noise = 0.5 * np.random.randn(N, 3)
        pts = _normalize(_fibonacci_points(N) + noise)
        initializations.append(pts)
        
    # Add pure random
    for _ in range(3):
        pts = np.random.randn(N, 3)
        pts = _normalize(pts)
        initializations.append(pts)

    for init_pts in initializations:
        if time.time() > deadline:
            break
            
        pts = init_pts.copy()
        
        # Stage 1: Global Repulsion Optimization
        # Flatten pts for scipy minimize
        x0 = pts.flatten()
        
        def objective(x):
            p = x.reshape(N, 3)
            p = _normalize(p)
            return _repulsion_energy(p)
            
        def gradient(x):
            p = x.reshape(N, 3)
            p = _normalize(p)
            g = _repulsion_gradient(p)
            return g.flatten()
            
        try:
            res = minimize(
                objective, 
                x0, 
                jac=gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            pts = _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
            
        # Stage 2: Local Polish for Covering Radius
        if time.time() > deadline:
            break
            
        pts = _polish_facets(pts, max_iter=100, step=0.2)
        
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    # Final polish on the best found
    if time.time() < deadline:
        best_pts = _polish_facets(best_pts, max_iter=200, step=0.1)
        best_cr = _covering_radius(best_pts)
        
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
