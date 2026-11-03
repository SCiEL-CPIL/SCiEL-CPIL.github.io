"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = len(pts)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal and upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _grad(pts):
    """Compute gradient of energy with respect to points on sphere.
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = sum_{j!=i} (xi-xj)/|xi-xj|^2
    Project to tangent space: grad - (grad.xi)xi
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    np.fill_diagonal(dists_sq, 1.0)
    # grad component: sum_j (xi-xj)/|xi-xj|^2
    # factor[i,j] = 1/|xi-xj|^2
    factor = 1.0 / dists_sq
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    # Project to tangent space
    for i in range(n):
        proj = np.dot(grad[i], pts[i])
        grad[i] -= proj * pts[i]
    return grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_cg(pts, max_iter=200, tol=1e-12):
    """Riemannian conjugate gradient on the sphere."""
    n = len(pts)
    x = _project_to_sphere(pts.copy())
    f = _energy(x)
    g = _grad(x)
    # Initial direction: negative gradient
    d = -g.copy()
    # Riemannian conjugate gradient
    for k in range(max_iter):
        # Check convergence
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            break
        
        # Line search: minimize f(x + t*d) along geodesic
        # Use simple backtracking
        t = 1.0
        best_t = 0.0
        best_f = f
        for ls in range(30):
            # Move along geodesic: x_new = x*cos(t) + d*sin(t)/|d|
            dnorm = np.linalg.norm(d)
            if dnorm < 1e-15:
                break
            d_unit = d / dnorm
            cos_t = np.cos(t * dnorm)
            sin_t = np.sin(t * dnorm)
            x_new = x * cos_t + d_unit * sin_t
            x_new = _project_to_sphere(x_new)
            f_new = _energy(x_new)
            if f_new < best_f:
                best_f = f_new
                best_t = t
            else:
                break
            t *= 0.5
        
        if best_t == 0.0:
            break
        
        dnorm = np.linalg.norm(d)
        d_unit = d / dnorm
        cos_t = np.cos(best_t * dnorm)
        sin_t = np.sin(best_t * dnorm)
        x_new = x * cos_t + d_unit * sin_t
        x_new = _project_to_sphere(x_new)
        
        g_new = _grad(x_new)
        f_new = best_f
        
        # Compute beta (Hestenes-Stiefel)
        # beta = <g_new, g_new> / <g, g>
        g_old = g
        g_new_norm_sq = np.sum(g_new * g_new)
        g_old_norm_sq = np.sum(g_old * g_old)
        if g_old_norm_sq > 1e-30:
            beta = g_new_norm_sq / g_old_norm_sq
        else:
            beta = 0.0
        
        # New direction
        d_new = -g_new + beta * d
        # Project d_new to tangent space
        for i in range(n):
            proj = np.dot(d_new[i], x_new[i])
            d_new[i] -= proj * x_new[i]
        
        x = x_new
        f = f_new
        g = g_new
        d = d_new
    
    return x, f

def _basin_hopping_step(pts, temperature=0.1, n_move=None):
    """Perform a basin hopping step: random perturbation + local optimization."""
    n = len(pts)
    if n_move is None:
        n_move = max(1, int(n * 0.1))
    
    # Random perturbation
    idx = np.random.choice(n, size=n_move, replace=False)
    for i in idx:
        # Random direction in tangent space
        rand_dir = np.random.randn(3)
        rand_dir -= np.dot(rand_dir, pts[i]) * pts[i]
        rand_dir_norm = np.linalg.norm(rand_dir)
        if rand_dir_norm > 1e-15:
            rand_dir /= rand_dir_norm
        angle = temperature * np.random.randn()
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        pts[i] = pts[i] * cos_a + rand_dir * sin_a
    
    pts = _project_to_sphere(pts)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Initial configuration: golden angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # First local optimization
    pts, f = _riemannian_cg(pts, max_iter=300)
    best_pts = pts.copy()
    best_f = f
    
    # Basin hopping loop
    np.random.seed(42)
    temperature = 0.15
    iteration = 0
    
    while time.time() < deadline:
        # Perturb
        perturbed = _basin_hopping_step(best_pts, temperature=temperature, n_move=max(3, int(N * 0.15)))
        
        # Local optimization
        optimized, f_new = _riemannian_cg(perturbed, max_iter=200)
        
        # Accept if better
        if f_new < best_f:
            best_f = f_new
            best_pts = optimized.copy()
            # Reduce temperature slowly
            temperature = max(0.05, temperature * 0.99)
        else:
            # Metropolis acceptance
            delta_f = f_new - best_f
            if delta_f < 0:
                accept = True
            else:
                accept = np.random.rand() < math.exp(-delta_f / (temperature * 10))
            if accept:
                best_pts = optimized.copy()
                best_f = f_new
            else:
                # Keep best, increase temperature slightly
                temperature = min(0.25, temperature * 1.01)
        
        iteration += 1
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final polish
    if time.time() < deadline - 5.0:
        best_pts, best_f = _riemannian_cg(best_pts, max_iter=500)
    
    return _project_to_sphere(best_pts)
# EVOLVE-BLOCK-END
