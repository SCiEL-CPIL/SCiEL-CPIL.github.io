"""Covering the sphere with 130 equal caps - soft surrogate gradient + exact hull polish."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import warnings

N = 130

def _covering_radius_from_hull(pts):
    """Compute exact covering radius in radians using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            v1 = a - b
            v2 = a - c
            c_dir = np.cross(v1, v2)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-14:
                continue
            c_dir = c_dir / c_norm
            if np.dot(c_dir, a) < 0:
                c_dir = -c_dir
            cos_r = np.clip(np.dot(c_dir, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        return np.pi

def _soft_covering_radius_and_grad(pts, beta=40.0):
    """
    Smooth approximation of covering radius via log-sum-exp over a dense grid.
    Returns (soft_radius, gradient w.r.t. pts projected to tangent space).
    """
    n_grid = 2000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = grid_pts @ pts.T  # (n_grid, N)
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    max_dot = np.max(dots, axis=1)  # (n_grid,)
    angles = np.arccos(max_dot)
    
    # Log-sum-exp soft max
    b_scaled = beta * angles
    b_max = np.max(b_scaled)
    exp_sum = np.sum(np.exp(b_scaled - b_max))
    soft_r = (b_max + np.log(exp_sum)) / beta
    
    # Weights for soft max
    w = np.exp(b_scaled - b_max) / exp_sum  # (n_grid,)
    
    # Softmax assignment weights per grid point
    d_scaled = beta * dots
    d_max = np.max(d_scaled, axis=1, keepdims=True)
    exp_d = np.exp(d_scaled - d_max)
    sm_w = exp_d / np.sum(exp_d, axis=1, keepdims=True)  # (n_grid, N)
    
    # d(angle)/d(max_dot) = -1/sqrt(1 - max_dot^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_ang = -1.0 / np.sqrt(np.maximum(1.0 - max_dot**2, 1e-12))
    
    # Gradient
    gf = w * d_ang  # (n_grid,)
    grad = gf @ sm_w  # (N,)
    
    # Project to tangent space
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    return soft_r, grad

def _repulsive_force_step(pts, step_size):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    w = 1.0 / dist_sq
    np.fill_diagonal(w, 0.0)
    forces = np.sum(w[:, :, np.newaxis] * diff, axis=1)
    pts_new = pts + step_size * forces
    norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts_new / norms

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _exact_local_polish(pts, start_time, time_limit=145.0, max_iters=500):
    best_pts = pts.copy()
    best_r = _covering_radius_from_hull(pts)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit - 3:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_facet = None
        worst_r = 0.0
        worst_cc = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            v1 = a - b
            v2 = a - c
            c_dir = np.cross(v1, v2)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-14:
                continue
            c_dir = c_dir / c_norm
            if np.dot(c_dir, a) < 0:
                c_dir = -c_dir
            cos_r = np.clip(np.dot(c_dir, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > worst_r:
                worst_r = r
                worst_facet = simplex
                worst_cc = c_dir
        
        if worst_facet is None:
            break
        
        improved = False
        for step in [0.03, 0.01, 0.003, 0.001]:
            pts_trial = pts.copy()
            for idx in worst_facet:
                pts_trial[idx] = pts_trial[idx] + step * (worst_cc - pts_trial[idx])
            pts_trial = _project_to_sphere(pts_trial)
            trial_r = _covering_radius_from_hull(pts_trial)
            if trial_r < best_r:
                pts = pts_trial
                best_r = trial_r
                best_pts = pts.copy()
                improved = True
                break
        
        if not improved:
            # Try random perturbation of all points
            if np.random.random() < 0.05:
                noise = np.random.randn(N, 3) * 0.005
                pts_trial = _project_to_sphere(pts + noise)
                trial_r = _covering_radius_from_hull(pts_trial)
                if trial_r < best_r:
                    pts = pts_trial
                    best_r = trial_r
                    best_pts = pts.copy()
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # 1. Fibonacci spiral init
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _covering_radius_from_hull(pts)
    
    # 2. Repulsive force phase
    step_size = 0.02
    cooling = 0.9995
    while time.time() - start_time < time_limit - 50:
        pts = _repulsive_force_step(pts, step_size)
        step_size *= cooling
        if int(step_size * 1e8) % 100 == 0:
            current_r = _covering_radius_from_hull(pts)
            if current_r < best_radius:
                best_radius = current_r
                best_pts = pts.copy()
        if step_size < 1e-9:
            break
    
    current_r = _covering_radius_from_hull(pts)
    if current_r < best_radius:
        best_radius = current_r
        best_pts = pts.copy()
    
    # 3. Soft surrogate gradient phase
    remaining = time_limit - (time.time() - start_time)
    if remaining > 25:
        lr = 0.005
        beta = 40.0
        while time.time() - start_time < time_limit - 15:
            soft_r, grad = _soft_covering_radius_and_grad(pts, beta=beta)
            
            improved = False
            step = lr
            for _ in range(4):
                pts_trial = _project_to_sphere(pts - step * grad)
                trial_r = _covering_radius_from_hull(pts_trial)
                if trial_r < best_radius:
                    pts = pts_trial
                    best_radius = trial_r
                    best_pts = pts.copy()
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                lr *= 0.6
                if lr < 1e-7:
                    break
                # Occasional random restart
                if np.random.random() < 0.05:
                    pts = _project_to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
    
    # 4. Exact local polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 8:
        best_pts, best_radius = _exact_local_polish(
            best_pts, start_time=start_time, time_limit=time_limit
        )
    
    return best_pts
