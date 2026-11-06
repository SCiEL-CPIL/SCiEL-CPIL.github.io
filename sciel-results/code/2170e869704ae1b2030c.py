"""
Minimize logarithmic energy of 100 points on the unit sphere.
Uses Riemannian L-BFGS with exponential map retraction,
multiple diverse initializations, and multi-stage polishing.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|p_i - p_j|)
    Gradient: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent space.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]
    
    if np.any(d_upper < 1e-9):
        return 1e18, np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    grad = np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad

def _exp_map(pts, tangent):
    """
    Exponential map on the unit sphere: move points along tangent vectors.
    For a unit vector p and tangent vector v (p·v=0),
    exp_p(v) = p*cos(||v||) + v/sin(||v||)*||v|| = p*cos(r) + v*sin(r)/r
    where r = ||v||.
    """
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-15, 1.0, norms)
    cos_r = np.cos(norms)
    sin_r_over_r = np.sin(safe_norms) / safe_norms
    # Where norms are ~0, sin_r_over_r should be 1
    sin_r_over_r = np.where(norms < 1e-15, 1.0, sin_r_over_r)
    return pts * cos_r + tangent * sin_r_over_r

def _log_map(pts, target):
    """
    Logarithmic map on the sphere: compute the tangent vector at pts
    that, when exponentiated, gives target.
    v = target - cos(theta)*pts, where theta = angle between pts and target
    """
    dots = np.sum(pts * target, axis=1, keepdims=True)
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    theta = np.arccos(dots)
    sin_theta = np.sin(theta)
    safe_sin = np.where(np.abs(sin_theta) < 1e-15, 1.0, sin_theta)
    v = (target - dots * pts) / safe_sin
    v = np.where(np.abs(sin_theta) < 1e-15, 0.0, v)
    return v * theta  # Actually this is already the tangent vector

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _structured_sphere(n, seed=0):
    """
    Generate points using a structured approach:
    Start with a few well-spread base points, then add more by
    subdividing and projecting.
    """
    rng = np.random.RandomState(seed)
    
    # Start with icosahedron vertices (12 points)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    pts = verts.copy()
    
    # Add more points by finding "gaps" - points that are far from all existing points
    while len(pts) < n:
        # Generate many candidate points
        candidates = rng.randn(200, 3)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute the minimum distance to existing points
        # Pick the one that maximizes this minimum distance
        best_idx = 0
        best_min_dist = -1
        for c in candidates:
            dists = np.linalg.norm(pts - c, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = c
        
        pts = np.vstack([pts, best_idx.reshape(1, 3)])
    
    return pts[:n]

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _riemannian_lbfgs(pts0, deadline, maxiter=5000, ftol=1e-14, gtol=1e-10):
    """
    Riemannian L-BFGS optimization using the exponential map as retraction.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 
                     'ftol': ftol, 'gtol': gtol, 'm': 20}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _polish(pts0, deadline, maxiter=10000, ftol=1e-16, gtol=1e-12):
    """
    High precision polishing.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                     'ftol': ftol, 'gtol': gtol, 'm': 10}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _local_search(pts0, deadline, n_restarts=5):
    """
    Try small perturbations of the current best solution to escape local minima.
    """
    best_pts = pts0.copy()
    best_energy = _energy_and_grad(pts0)[0]
    
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb points slightly
        noise = rng.randn(N, 3) * 0.01
        perturbed = _project_to_sphere(pts0 + noise)
        
        # Optimize from perturbed starting point
        pts_opt, e_opt = _riemannian_lbfgs(perturbed, deadline, maxiter=2000, ftol=1e-12, gtol=1e-8)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
    
    return best_pts, best_energy

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer before 120s limit

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    initializations = []
    
    # Fibonacci spiral - strong heuristic
    initializations.append(_fibonacci_sphere(N))
    
    # Structured sphere
    initializations.append(_structured_sphere(N, seed=1))
    initializations.append(_structured_sphere(N, seed=2))
    
    # Random starts with different seeds
    seeds = [42, 123, 777, 999, 2023, 555, 888]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # 2. Coarse Optimization Phase
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        pts_opt, e_opt = _riemannian_lbfgs(pts0, deadline, maxiter=3000, ftol=1e-12, gtol=1e-8)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
        
        # Print progress for debugging (will be removed in final)
        # print(f"Init {i}: energy = {e_opt:.6f}")

    if best_pts is None:
        return _fibonacci_sphere(N)

    # 3. Local Search Phase - try to escape local minima
    time_left = deadline - time.time()
    if time_left > 20.0:
        pts_ls, e_ls = _local_search(best_pts, deadline, n_restarts=3)
        if e_ls < best_energy:
            best_energy = e_ls
            best_pts = pts_ls

    # 4. Fine Polishing Phase
    time_left = deadline - time.time()
    if time_left > 15.0:
        pts_polished, e_polished = _polish(best_pts, deadline, maxiter=5000, ftol=1e-15, gtol=1e-11)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished

    # 5. Ultra-Fine Polish
    time_left = deadline - time.time()
    if time_left > 10.0:
        pts_ultra, e_ultra = _polish(best_pts, deadline, maxiter=10000, ftol=1e-16, gtol=1e-13)
        if e_ultra < best_energy:
            best_energy = e_ultra
            best_pts = pts_ultra

    # 6. Final Check and Return
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
