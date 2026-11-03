import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150
DEADLINE_MARGIN = 10.0  # seconds before hard limit

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_and_grad(pts):
    """Compute energy and gradient (Cartesian) for points on sphere.
    Returns (energy, grad) where grad is (N,3)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    energy = -0.5 * np.sum(np.log(np.sqrt(dist_sq[iu])))
    
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = -sum_{j != i} (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _subdivided_icosahedron(n_points):
    """Generate approximately uniform points on sphere via icosahedron subdivision."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide until we have at least n_points vertices
    while len(verts) < n_points:
        new_faces = []
        midpoint_cache = {}
        
        def get_midpoint(i1, i2):
            key = (min(i1, i2), max(i1, i2))
            if key in midpoint_cache:
                return midpoint_cache[key]
            mid = (verts[i1] + verts[i2]) / 2.0
            mid = mid / np.linalg.norm(mid)
            idx = len(verts)
            verts = np.vstack([verts, mid])
            midpoint_cache[key] = idx
            return idx
        
        for f in faces:
            a, b, c = f
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        
        faces = new_faces
    
    # Use first n_points vertices
    pts = verts[:n_points]
    return _project_to_sphere(pts)

def _random_uniform_sphere(n, seed=None):
    """Generate n random uniform points on the sphere."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1 - z**2)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _tangent_vector(v, p):
    """Project vector v onto tangent space at point p on unit sphere."""
    return v - np.dot(v, p) * p

def _soft_min_surrogate(pts, temperature=0.01):
    """
    Soft-min surrogate for the logarithmic energy.
    Instead of summing log(1/d) directly, we use a smooth approximation
    that avoids the singularity at d=0.
    
    Uses: log(1/d) ≈ -log(d + eps) where eps is small,
    but more importantly, we add a repulsive term that's smooth.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    
    # Soft approximation: use log(d + eps) instead of log(d)
    # This makes the energy finite and smooth everywhere
    eps = temperature
    soft_energy = -0.5 * np.sum(np.log(dist[iu] + eps))
    
    # Gradient of soft energy
    # d/dx_i [-log(|x_i - x_j| + eps)] = -(x_i - x_j) / ((|x_i - x_j| + eps) * |x_i - x_j|)
    inv_dist = 1.0 / dist
    np.fill_diagonal(inv_dist, 0.0)
    weight = inv_dist / (dist + eps)
    np.fill_diagonal(weight, 0.0)
    grad = -np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    return soft_energy, grad

def _gradient_descent_sphere(pts, max_iter=500, tol=1e-10, deadline=None):
    """Geodesic gradient descent on the sphere with adaptive step size."""
    pts = pts.copy()
    lr = 1e-3
    best_e = _log_energy(pts)
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # Adaptive step size with backtracking
        for attempt in range(20):
            step = lr * (1.0 / (1.0 + g_norm))
            new_pts = _project_to_sphere(pts - step * g_tan)
            new_e = _log_energy(new_pts)
            
            if new_e < e - 1e-12:
                pts = new_pts
                e = new_e
                if e < best_e:
                    best_e = e
                break
            else:
                step *= 0.5
        
        # Adjust learning rate
        if k % 50 == 0 and k > 0:
            lr = min(lr * 1.1, 0.05)
    
    return pts, best_e

def _basin_hopping_single(pts, deadline, n_iter=30):
    """Basin hopping: perturb, then minimize."""
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        # Random perturbation in tangent space
        rng = np.random.default_rng(i)
        perturb = rng.normal(0, 0.1, (N, 3))
        perturb = perturb - np.sum(perturb * best_pts, axis=1, keepdims=True) * best_pts
        perturbed = _project_to_sphere(best_pts + perturb)
        
        # Minimize from perturbed position
        polished, e = _gradient_descent_sphere(perturbed, max_iter=200, tol=1e-9, deadline=deadline)
        
        if e < best_e:
            best_e = e
            best_pts = polished
    
    return best_pts, best_e

def _high_precision_polish(pts, deadline):
    """Final high-precision polish using L-BFGS-B on spherical coordinates."""
    # Convert to spherical coordinates
    z = pts[:, 2]
    z = np.clip(z, -1 + 1e-12, 1 - 1e-12)
    theta = np.arccos(z)
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    
    x0 = np.concatenate([theta, phi])
    
    def obj_and_grad(x):
        th = x[:N]
        ph = x[N:]
        p = np.column_stack([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th)
        ])
        e, g_cart = _energy_and_grad(p)
        
        # Spherical basis
        dth = np.column_stack([
            np.cos(th) * np.cos(ph),
            np.cos(th) * np.sin(ph),
            -np.sin(th)
        ])
        dph = np.column_stack([
            -np.sin(th) * np.sin(ph),
            np.sin(th) * np.cos(ph),
            np.zeros(N)
        ])
        
        g_th = np.sum(g_cart * dth, axis=1)
        g_ph = np.sum(g_cart * dph, axis=1)
        
        return e, np.concatenate([g_th, g_ph])
    
    bounds = [(1e-8, np.pi - 1e-8)] * N + [(0, 2 * np.pi)] * N
    
    try:
        res = minimize(
            fun=obj_and_grad,
            x0=x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if res.success or res.fun < _log_energy(pts):
            th_opt, ph_opt = res.x[:N], res.x[N:]
            pts_new = np.column_stack([
                np.sin(th_opt) * np.cos(ph_opt),
                np.sin(th_opt) * np.sin(ph_opt),
                np.cos(th_opt)
            ])
            return pts_new
    except Exception:
        pass
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Generate multiple initial candidates
    candidates = []
    
    # 1. Subdivided icosahedron
    candidates.append(_subdivided_icosahedron(N))
    
    # 2. Random uniform (multiple seeds)
    for seed in [42, 123, 456]:
        candidates.append(_random_uniform_sphere(N, seed=seed))
    
    # 3. Icosahedron with small noise
    base = _subdivided_icosahedron(N)
    rng = np.random.default_rng(789)
    noise = rng.normal(0, 0.05, (N, 3))
    noise = noise - np.sum(noise * base, axis=1, keepdims=True) * base
    candidates.append(_project_to_sphere(base + noise))
    
    best_pts = None
    best_energy = np.inf
    
    # Phase 1: Gradient descent from each candidate
    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
        
        polished, e = _gradient_descent_sphere(init_pts, max_iter=400, tol=1e-10, deadline=deadline - 30.0)
        
        if e < best_energy:
            best_energy = e
            best_pts = polished
    
    # Phase 2: Basin hopping from best
    if best_pts is not None and time.time() < deadline - 15.0:
        best_pts, best_energy = _basin_hopping_single(best_pts, deadline - 15.0, n_iter=20)
    
    # Phase 3: High-precision polish
    if best_pts is not None and time.time() < deadline - 5.0:
        best_pts = _high_precision_polish(best_pts, deadline)
    
    # Final gradient descent polish
    if best_pts is not None:
        best_pts, _ = _gradient_descent_sphere(best_pts, max_iter=200, tol=1e-12, deadline=deadline)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to icosahedron
        best_pts = _subdivided_icosahedron(N)
    
    # Verify all values are finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _subdivided_icosahedron(N)
    
    return best_pts
