"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.85  # safety margin
TIME_LIMIT = 150.0


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)."""
    n = pts.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return -1e18
            total += math.log(1.0 / d)
    return total


def _log_energy_fast(pts):
    """Vectorized log energy."""
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only use upper triangle
    iu = np.triu_indices(n, k=1)
    return np.sum(np.log(1.0 / dists[iu]))


def _log_energy_grad(pts):
    """Compute log energy and its gradient.
    
    Energy E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2
    """
    n = pts.shape[0]
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-30:
                return -1e18, grad
            d = math.sqrt(d2)
            energy += math.log(1.0 / d)
            # gradient contribution
            factor = 1.0 / d2
            grad[i] += diff * factor
            grad[j] -= diff * factor
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(pts, deadline):
    """Run L-BFGS-B optimization from initial points, respecting deadline."""
    pts = _project_to_sphere(pts)
    
    # Flatten to 1D for scipy
    x0 = pts.flatten()
    
    def objective_and_grad(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        e, g = _log_energy_grad(p)
        return e, g.flatten()
    
    # Use L-BFGS-B with gradient
    result = None
    best_x = x0.copy()
    best_e = _log_energy_fast(_project_to_sphere(x0.reshape(-1, 3)))
    
    try:
        def func(x):
            p = _project_to_sphere(x.reshape(-1, 3))
            return _log_energy_fast(p)
        
        def jac(x):
            p = _project_to_sphere(x.reshape(-1, 3))
            _, g = _log_energy_grad(p)
            return g.flatten()
        
        # Multiple restarts with different step sizes / approaches
        options = {
            'maxiter': 5000,
            'gtol': 1e-12,
            'ftol': 1e-15,
            'disp': False,
            'maxfun': 10000,
        }
        
        # Check time
        elapsed = time.time() - _start_time
        if elapsed > deadline:
            return best_x.reshape(-1, 3)
        
        result = minimize(
            func, x0, jac=jac,
            method='L-BFGS-B',
            options=options
        )
        
        if result.fun < best_e:
            best_x = result.x
            best_e = result.fun
        
    except Exception:
        pass
    
    # Polish with steepest descent on the manifold
    pts = _project_to_sphere(best_x.reshape(-1, 3))
    pts = _polish(pts, deadline)
    
    return pts


def _polish(pts, deadline):
    """Steepest descent polish on the sphere manifold."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    lr = 0.1
    min_lr = 1e-10
    patience = 0
    
    while time.time() - _start_time < deadline:
        e, g = _log_energy_grad(pts)
        if e < -1e17:
            break
        
        # Project gradient onto tangent space
        pts_t = pts.copy()
        g_t = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        
        # Check convergence
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-14:
            break
        
        # Step
        step = lr * g_t
        new_pts = pts + step
        new_pts = _project_to_sphere(new_pts)
        
        new_e = _log_energy_fast(new_pts)
        
        if new_e < current_energy:
            pts = new_pts
            current_energy = new_e
            patience = 0
            lr = min(lr * 1.05, 1.0)
        else:
            lr *= 0.5
            patience += 1
            if patience > 30 or lr < min_lr:
                break
    
    return pts


def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    # Start from Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Optimize
    pts = _optimize_from_start(pts, deadline)
    
    # Final check and normalization
    pts = _project_to_sphere(pts)
    
    # Verify validity
    if not np.all(np.isfinite(pts)):
        return _fibonacci_spiral(N)
    
    # Check minimum distance
    min_d = 1e18
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-9:
        return _fibonacci_spiral(N)
    
    return pts
# EVOLVE-BLOCK-END
