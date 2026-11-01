"""Sphere covering with 130 equal caps - combined approach."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import warnings

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius in degrees using convex hull facets."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    norms[norms < 1e-12] = 1e-12
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return _covering_radius_brute(points)
    
    max_circ = 0.0
    for face in faces:
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        r1 = np.arccos(np.clip(n @ p1, -1, 1))
        r2 = np.arccos(np.clip(-n @ p1, -1, 1))
        r = min(r1, r2)
        max_circ = max(max_circ, r)
    
    return np.degrees(max_circ)


def _covering_radius_brute(points, n_theta=180, n_phi=360):
    """Brute force covering radius computation on a grid."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    norms[norms < 1e-12] = 1e-12
    points = points / norms[:, None]
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid), chunk_size):
        end = min(start + chunk_size, len(grid))
        g = grid[start:end]
        dots = g @ points.T
        dots = np.clip(dots, -1, 1)
        max_dots = np.max(dots, axis=1)
        min_angs = np.arccos(max_dots)
        max_dist = max(max_dist, np.max(min_angs))
    
    return np.degrees(max_dist)


def _fibonacci_spiral(n):
    """Generate points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(points, axis=1)
    return points / norms[:, None]


def _lloyd_iteration(points, n_iter=50, tol=1e-10):
    """Lloyd's algorithm for spherical k-means / Voronoi centroid refinement."""
    points = np.array(points, dtype=float)
    n = len(points)
    
    for it in range(n_iter):
        n_theta = 360
        n_phi = 720
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(TH) * np.cos(PH)
        y = np.sin(TH) * np.sin(PH)
        z = np.cos(TH)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        dots = grid @ points.T
        nearest = np.argmax(dots, axis=1)
        
        new_points = np.zeros_like(points)
        counts = np.zeros(n)
        for i in range(n):
            mask = nearest == i
            if np.sum(mask) > 0:
                new_points[i] = grid[mask].mean(axis=0)
                counts[i] = np.sum(mask)
        
        for i in range(n):
            if counts[i] == 0:
                new_points[i] = points[i]
        
        norms = np.linalg.norm(new_points, axis=1)
        norms[norms < 1e-12] = 1e-12
        new_points = new_points / norms[:, None]
        
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < tol:
            break
    
    return points


def _facet_polish(points, max_time=30.0):
    """Polish by iteratively moving worst-facet vertices toward their circumcenter."""
    points = np.array(points, dtype=float)
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    lr = 0.05
    no_improve = 0
    
    while time.time() - start < max_time and no_improve < 50:
        norms = np.linalg.norm(points, axis=1)
        norms[norms < 1e-12] = 1e-12
        points = points / norms[:, None]
        
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
        except Exception:
            break
        
        max_circ = 0.0
        worst_face = None
        for face in faces:
            p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            r1 = np.arccos(np.clip(n @ p1, -1, 1))
            r2 = np.arccos(np.clip(-n @ p1, -1, 1))
            r = min(r1, r2)
            if r > max_circ:
                max_circ = r
                worst_face = face
        
        if worst_face is None:
            break
        
        p1, p2, p3 = points[worst_face[0]], points[worst_face[1]], points[worst_face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            break
        n = n / n_norm
        c = n if (n @ p1) > 0 else -n
        
        for idx in worst_face:
            p = points[idx]
            d = c - p
            d = d - (d @ p) * p
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-12:
                d = d / d_norm
                points[idx] = points[idx] + lr * d
        
        norms = np.linalg.norm(points, axis=1)
        norms[norms < 1e-12] = 1e-12
        points = points / norms[:, None]
        
        new_radius = _covering_radius(points)
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            points = best_points.copy()
            lr *= 0.85
    
    return best_points


def _soft_max_loss(x_flat, grid_pts, temperature=30.0):
    """
    Compute a smooth approximation of the covering radius using soft-max over
    soft-min distances. Returns (value, gradient) with respect to x_flat.
    """
    pts = x_flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    pts = pts / norms
    
    M = len(grid_pts)
    
    # Angular distances: M x N
    dot = grid_pts @ pts.T
    dot = np.clip(dot, -1.0, 1.0)
    ang_dists = np.arccos(dot)
    
    # Soft-min over points for each grid point
    T_min = 30.0
    exp_neg = np.exp(-T_min * ang_dists)
    sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
    sum_exp[sum_exp < 1e-300] = 1e-300
    soft_min = -1.0 / T_min * np.log(sum_exp)[:, 0]
    
    # Weights for soft-min
    w = exp_neg / sum_exp  # M x N
    
    # Soft-max over grid points
    T_max = 30.0
    max_sm = np.max(soft_min)
    shifted = soft_min - max_sm
    exp_shifted = np.exp(T_max * shifted)
    sum_exp_max = np.sum(exp_shifted)
    if sum_exp_max < 1e-300:
        sum_exp_max = 1e-300
    val = max_sm + (1.0 / T_max) * np.log(sum_exp_max)
    
    # Gradient of soft-max w.r.t soft_min
    w_max = exp_shifted / sum_exp_max  # M
    
    # Gradient of soft_min w.r.t pts
    # d(soft_min[j])/d(pts[k]) = sum_k' w[j,k'] * d(ang_dists[j,k'])/d(pts[k'])
    # d(arccos(u))/d(pts[k']) = -u_k / sqrt(1-u^2) where u_k = grid[j] dot pts[k]
    # But we need to be careful: d(arccos(u))/dx = -1/sqrt(1-u^2) * du/dx
    # du/dx where u = g dot p, so du/dp_k = g
    # d(arccos(u))/dp_k = -g / sqrt(1 - u^2)
    
    # Compute gradient efficiently
    # For each grid point j and each point k:
    # contrib[j,k] = w_max[j] * w[j,k] * (-grid_pts[j] / sqrt(1 - dot[j,k]^2))
    
    # Vectorized:
    sqrt_term = np.sqrt(np.clip(1.0 - dot * dot, 1e-10, 1.0))  # M x N
    # grad_pts[k] = sum_j w_max[j] * w[j,k] * (-grid_pts[j] / sqrt_term[j,k])
    
    # Build the gradient matrix
    # G[j,k] = w_max[j] * w[j,k] / sqrt_term[j,k]
    G = w_max[:, np.newaxis] * w / sqrt_term  # M x N
    
    # grad_pts[k] = -sum_j G[j,k] * grid_pts[j]
    grad_pts = -(G.T @ grid_pts)  # N x 3
    
    # Project gradient onto tangent space (remove radial component)
    grad_flat = grad_pts.reshape(-1)
    
    return float(val), grad_flat


def _gradient_descent_smooth(pts, grid_pts, max_iter=200, alpha_init=0.01, decay=0.98, temperature=30.0):
    """
    Gradient descent on the smooth soft-max loss to minimize covering radius.
    """
    x = pts.reshape(-1).copy()
    best_x = x.copy()
    best_val = _covering_radius(pts)
    
    alpha = alpha_init
    for it in range(max_iter):
        val, grad = _soft_max_loss(x, grid_pts, temperature)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        x = x - alpha * grad
        
        # Project back to sphere
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        p = p / norms
        
        # Check exact covering radius
        exact_r = _covering_radius(p)
        if exact_r < best_val:
            best_val = exact_r
            best_x = x.copy()
        
        alpha *= decay
        
        # Adaptive temperature: decrease as we converge
        temperature = max(10.0, temperature * 0.995)
    
    p = best_x.reshape(N, 3)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    return p / norms


def _repulsion_refine(pts, max_iter=100, alpha_init=0.005):
    """
    Refine using repulsion energy gradient to spread points more evenly.
    """
    x = pts.copy()
    alpha = alpha_init
    
    for it in range(max_iter):
        dot = x @ x.T
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** 2)
        np.fill_diagonal(w, 0.0)
        
        w_grad = 2.0 * (dist_sq ** (-1.5))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ x
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        x = x + alpha * grad / grad_norm
        
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        x = x / norms
        
        alpha *= 0.99
    
    return x


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Stage 1: Fibonacci spiral initialization
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Stage 2: Lloyd's algorithm for global balance
    if time.time() < deadline - 90:
        lloyd_points = _lloyd_iteration(points, n_iter=60)
        lloyd_radius = _covering_radius(lloyd_points)
        if lloyd_radius < best_radius:
            best_points = lloyd_points
            best_radius = lloyd_radius
    
    # Stage 3: Smooth gradient descent on soft-max loss
    if time.time() < deadline - 50:
        # Create a reasonably dense grid for the smooth objective
        n_theta = 200
        n_phi = 400
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        gx = np.sin(TH) * np.cos(PH)
        gy = np.sin(TH) * np.sin(PH)
        gz = np.cos(TH)
        grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        
        smooth_points = _gradient_descent_smooth(best_points, grid, max_iter=150, alpha_init=0.008, decay=0.985)
        smooth_radius = _covering_radius(smooth_points)
        if smooth_radius < best_radius:
            best_points = smooth_points
            best_radius = smooth_radius
    
    # Stage 4: Repulsion-based refinement
    if time.time() < deadline - 30:
        rep_points = _repulsion_refine(best_points, max_iter=80)
        rep_radius = _covering_radius(rep_points)
        if rep_radius < best_radius:
            best_points = rep_points
            best_radius = rep_radius
    
    # Stage 5: Facet-based polishing
    if time.time() < deadline - 15:
        polished = _facet_polish(best_points, max_time=min(25, deadline - time.time() - 5))
        polish_radius = _covering_radius(polished)
        if polish_radius < best_radius:
            best_points = polished
            best_radius = polish_radius
    
    # Stage 6: Final smooth refinement if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        n_theta = 200
        n_phi = 400
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        gx = np.sin(TH) * np.cos(PH)
        gy = np.sin(TH) * np.sin(PH)
        gz = np.cos(TH)
        grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        
        max_iter = int((deadline - time.time() - 5) * 8)
        max_iter = max(50, min(max_iter, 200))
        smooth2 = _gradient_descent_smooth(best_points, grid, max_iter=max_iter, alpha_init=0.005, decay=0.99)
        s2_radius = _covering_radius(smooth2)
        if s2_radius < best_radius:
            best_points = smooth2
            best_radius = s2_radius
        
        if time.time() < deadline - 10:
            polished2 = _facet_polish(best_points, max_time=min(10, deadline - time.time() - 3))
            p2_radius = _covering_radius(polished2)
            if p2_radius < best_radius:
                best_points = polished2
                best_radius = p2_radius
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1)
    norms[norms < 1e-12] = 1e-12
    best_points = best_points / norms[:, None]
    
    return best_points

# EVOLVE-BLOCK-END
