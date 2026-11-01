"""Minimal logarithmic energy of 100 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # safety margin before 150s kill
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _energy_and_grad(points):
    """Compute logarithmic energy and gradient."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    d_safe = np.where(mask, dist, 1.0)
    energy = -0.5 * np.sum(np.log(d_safe[mask]))
    
    inv_dist2 = np.zeros_like(dist)
    inv_dist2[mask] = 1.0 / (dist[mask] ** 2)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _energy_fsum(points):
    """Compute energy using math.fsum for accuracy."""
    n = points.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d > 1e-15:
                terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _riemannian_cg(points, max_iter=500, tol=1e-10):
    """Riemannian conjugate gradient on the sphere."""
    x = _project_to_sphere(points.copy())
    
    f, g = _energy_and_grad(x)
    gx = np.sum(g * x, axis=1, keepdims=True)
    g_t = g - gx * x
    
    d = -g_t
    dx = np.sum(d * x, axis=1, keepdims=True)
    d = d - dx * x
    
    best_x = x.copy()
    best_f = f
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        alpha = 0.01
        x_new = x + alpha * d
        x_new = _project_to_sphere(x_new)
        f_new, g_new = _energy_and_grad(x_new)
        
        max_ls = 30
        for ls in range(max_ls):
            if f_new < f:
                break
            alpha *= 0.5
            x_new = x + alpha * d
            x_new = _project_to_sphere(x_new)
            f_new, g_new = _energy_and_grad(x_new)
        
        if f_new < f:
            x = x_new
            f = f_new
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        gx_new = np.sum(g_new * x, axis=1, keepdims=True)
        g_t_new = g_new - gx_new * x
        grad_norm = np.linalg.norm(g_t_new)
        if grad_norm < tol:
            break
        
        d_old = d.copy()
        beta = np.sum(g_t_new * (g_t_new - g_t)) / (np.sum(g_t**2) + 1e-15)
        beta = max(0.0, beta)
        d = -g_t_new + beta * d_old
        dx = np.sum(d * x, axis=1, keepdims=True)
        d = d - dx * x
        
        g_t = g_t_new
    
    return _project_to_sphere(best_x)

def _lbfgs_polish(points, max_iter=2000, tol=1e-12):
    """L-BFGS-B polishing with penalty for sphere constraint."""
    x0 = points.flatten().copy()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e4 * np.sum((norms - 1.0)**2)
        f, _ = _energy_and_grad(_project_to_sphere(pts))
        return f + penalty
    
    def gradient(x):
        pts = x.reshape(-1, 3)
        f, g = _energy_and_grad(_project_to_sphere(pts))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        penalty_grad = 2e4 * (norms - 1.0) * pts / (norms + 1e-15)
        return (g + penalty_grad).flatten()
    
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol,
                            'maxls': 50})
    
    if res.success or res.fun < _energy_fsum(_project_to_sphere(x0.reshape(-1,3))):
        return _project_to_sphere(res.x.reshape(-1, 3))
    return _project_to_sphere(points)

def _fibonacci_spiral():
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(seed):
    """Random points on sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)

def _icosahedral_init():
    """Icosahedral symmetric initialization (approximate)."""
    pts = _fibonacci_spiral()
    rng = np.random.default_rng(42)
    pts += 0.01 * rng.standard_normal((N, 3))
    return _project_to_sphere(pts)

def _perturbed_init(base, seed, scale=0.02):
    """Perturb a base configuration."""
    rng = np.random.default_rng(seed)
    pts = base + scale * rng.standard_normal((N, 3))
    return _project_to_sphere(pts)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = _fibonacci_spiral()
    best_energy = _energy_fsum(best_points)
    
    # Try multiple initializations
    inits = [
        ("fib", _fibonacci_spiral()),
        ("rand1", _random_init(1)),
        ("rand2", _random_init(2)),
        ("rand3", _random_init(3)),
        ("rand4", _random_init(4)),
        ("rand5", _random_init(5)),
        ("rand6", _random_init(6)),
        ("rand7", _random_init(7)),
        ("rand8", _random_init(8)),
        ("icos", _icosahedral_init()),
    ]
    
    for name, init_pts in inits:
        if _time_remaining() < 15.0:
            break
        
        # Riemannian CG refinement
        refined = _riemannian_cg(init_pts, max_iter=300, tol=1e-10)
        e = _energy_fsum(refined)
        
        if e < best_energy:
            best_energy = e
            best_points = refined.copy()
        
        # L-BFGS polishing
        if _time_remaining() < 10.0:
            break
        
        polished = _lbfgs_polish(refined, max_iter=1000, tol=1e-12)
        e2 = _energy_fsum(polished)
        
        if e2 < best_energy:
            best_energy = e2
            best_points = polished.copy()
    
    # Try perturbed versions of good solutions
    if _time_remaining() > 30.0:
        for pseed in [101, 102, 103, 104, 105]:
            if _time_remaining() < 15.0:
                break
            perturbed = _perturbed_init(best_points, pseed, scale=0.03)
            refined_p = _riemannian_cg(perturbed, max_iter=300, tol=1e-11)
            e_p = _energy_fsum(refined_p)
            if e_p < best_energy:
                best_energy = e_p
                best_points = refined_p.copy()
                polished_p = _lbfgs_polish(refined_p, max_iter=1000, tol=1e-14)
                e_p2 = _energy_fsum(polished_p)
                if e_p2 < best_energy:
                    best_energy = e_p2
                    best_points = polished_p.copy()
    
    # Final high-precision polish on best
    if _time_remaining() > 15.0:
        final = _lbfgs_polish(best_points, max_iter=3000, tol=1e-15)
        e_final = _energy_fsum(final)
        if e_final < best_energy:
            best_points = final
            best_energy = e_final
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-9:
                best_points[j] += 1e-8 * (best_points[j] - best_points[i])
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
