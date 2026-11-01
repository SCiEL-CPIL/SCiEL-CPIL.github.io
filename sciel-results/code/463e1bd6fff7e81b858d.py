"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START) - 2.0  # 2s safety margin

def _energy(pts):
    """Logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    dsq = dist_sq[iu]
    dsq = np.maximum(dsq, 1e-30)
    # Use math.fsum for precision
    # Convert to list of floats for fsum
    return math.fsum(math.log(1.0 / math.sqrt(v)) for v in dsq)

def _energy_grad(pts):
    """Gradient of log(1/d) energy w.r.t. each point (in R^3). Vectorized."""
    n = len(pts)
    # diff[i,j] = pts[i] - pts[j], shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-30)
    # grad[i] = sum_j -diff[i,j] / dist_sq[i,j]
    # = -sum_j diff[i,j] / dist_sq[i,j]
    inv_dsq = 1.0 / dist_sq  # (n, n)
    inv_dsq[np.diag_indices(n)] = 0.0
    grad = -np.sum(diff * inv_dsq[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _sphere_fun_grad(x_flat):
    """Objective and gradient for L-BFGS on flat vector; projects to sphere."""
    pts = _project_to_sphere(x_flat.reshape(N, 3))
    e = _energy(pts)
    g = _energy_grad(pts)
    # Project gradient onto tangent space: g_tangent = g - (g.x)x
    pts_t = pts.T  # (3, N)
    g_t = g.T  # (3, N)
    dot = np.sum(pts_t * g_t, axis=0)  # (N,)
    g_tangent = g - dot[:, None] * pts
    return e, g_tangent.flatten()

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _lbfgs_optimize(pts, max_iter=200, gtol=1e-8):
    """Run L-BFGS-B on the sphere using projected gradients."""
    x0 = pts.flatten()
    best_e = _energy(pts)
    best_pts = pts.copy()
    
    for attempt in range(3):
        if _time_left() < 5.0:
            break
        def fun_grad(x):
            e, g = _sphere_fun_grad(x)
            return e, g
        try:
            res = minimize(fun_grad, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': gtol,
                                   'maxfun': max_iter * 3})
            if res.success or res.fun < best_e:
                new_pts = _project_to_sphere(res.x.reshape(N, 3))
                new_e = _energy(new_pts)
                if new_e < best_e:
                    best_e = new_e
                    best_pts = new_pts
                x0 = res.x
        except Exception:
            break
    return best_pts, best_e

def _riemannian_descent(pts, max_iter=1000, c1=1e-4, rho=0.5):
    """Riemannian Gradient Descent with Armijo line search."""
    best_e = _energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        g = _energy_grad(pts)
        # Project to tangent space
        dot = np.sum(pts * g, axis=1, keepdims=True)
        g_tan = g - dot * pts
        
        # Check convergence
        if np.linalg.norm(g_tan) < 1e-10:
            break
            
        # Line search
        step = 0.1
        cur_e = _energy(pts)
        for trial in range(20):
            # Riemannian exponential map (approx for small steps: project)
            new_pts = _project_to_sphere(pts - step * g_tan)
            new_e = _energy(new_pts)
            if new_e <= cur_e - c1 * step * np.sum(g_tan * g_tan):
                pts = new_pts
                cur_e = new_e
                if cur_e < best_e:
                    best_e = cur_e
                    best_pts = pts.copy()
                break
            step *= rho
        else:
            # If no step accepted, try a smaller fixed step or break
            if step < 1e-8:
                break
    return best_pts, best_e

def _simulated_annealing(pts, t_start=0.5, t_end=0.001, max_iter=2000):
    """Simulated annealing with rotation perturbations on the sphere."""
    best_e = _energy(pts)
    best_pts = pts.copy()
    cur_e = best_e
    rng = np.random.RandomState(123)
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        t = t_start * (t_end / t_start) ** (it / max_iter)
        # Adaptive perturbation size: larger when hot
        scale = 0.05 + 0.15 * (t / t_start)
        # Pick 1-5 random points to perturb
        n_perturb = rng.randint(1, 6)
        idx = rng.choice(N, n_perturb, replace=False)
        new_pts = pts.copy()
        for i in idx:
            # Generate a random rotation axis and angle
            axis = rng.randn(3)
            axis /= np.linalg.norm(axis)
            angle = rng.randn() * scale
            # Rodrigues' rotation formula
            v = pts[i]
            v_dot_k = np.dot(v, axis)
            v_cross_k = np.cross(v, axis)
            v_new = v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1 - np.cos(angle))
            new_pts[i] = v_new / np.linalg.norm(v_new)
        new_e = _energy(new_pts)
        delta = new_e - cur_e
        if delta < 0 or (t > 1e-10 and rng.rand() < math.exp(-delta / t)):
            pts = new_pts
            cur_e = new_e
            if cur_e < best_e:
                best_e = cur_e
                best_pts = pts.copy()
    return best_pts, best_e

def construct():
    global _START
    _START = time.time()
    
    # Phase 1: Start with Fibonacci spiral and optimize
    pts = _fibonacci_spiral()
    pts, e = _lbfgs_optimize(pts, max_iter=300, gtol=1e-8)
    
    # Phase 2: Riemannian descent polish
    pts, e = _riemannian_descent(pts, max_iter=1000)
    
    # Phase 3: More L-BFGS iterations with tighter tolerance
    if _time_left() > 10.0:
        pts, e = _lbfgs_optimize(pts, max_iter=500, gtol=1e-10)
    
    # Phase 4: Simulated annealing to escape local minima
    if _time_left() > 15.0:
        pts_sa, e_sa = _simulated_annealing(pts, t_start=0.3, t_end=0.0005, max_iter=5000)
        if e_sa < e:
            pts, e = pts_sa, e_sa
            # Polish after SA
            if _time_left() > 5.0:
                pts, e = _lbfgs_optimize(pts, max_iter=300, gtol=1e-10)
                pts, e = _riemannian_descent(pts, max_iter=500)
    
    # Phase 5: Random restarts if time remains
    if _time_left() > 20.0:
        # Use diverse initial guesses
        initial_guesses = []
        initial_guesses.append(_random_points(7))
        initial_guesses.append(_random_points(13))
        initial_guesses.append(_fibonacci_spiral(0.1))
        initial_guesses.append(_fibonacci_spiral(0.5))
        
        for pts_r in initial_guesses:
            if _time_left() < 10.0:
                break
            pts_r, e_r = _lbfgs_optimize(pts_r, max_iter=200, gtol=1e-8)
            if e_r < e:
                pts, e = pts_r, e_r
                pts, e = _riemannian_descent(pts, max_iter=500)
                if _time_left() > 5.0:
                    pts, e = _lbfgs_optimize(pts, max_iter=200, gtol=1e-10)
    
    # Final high-precision polish with L-BFGS at very tight tolerance
    if _time_left() > 5.0:
        pts, e = _lbfgs_optimize(pts, max_iter=500, gtol=1e-12)
    
    # Final Riemannian descent polish
    if _time_left() > 3.0:
        pts, e = _riemannian_descent(pts, max_iter=3000)
    
    # Ensure all points are on unit sphere
    pts = _project_to_sphere(pts)
    
    # Validate: no points too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                # Nudge apart
                pts[j] = _project_to_sphere(pts[j] + 1e-8 * np.random.randn(3))[0]
    
    return pts
# EVOLVE-BLOCK-END
