"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s kill


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e18
            terms.append(-math.log(d))
    return math.fsum(terms)


def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points on sphere (projected)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i-x_j|) = -(x_i - x_j)/|x_i-x_j|^2
            factor = -1.0 / d
            grad[i] += factor * diff
            grad[j] -= factor * diff
    # Project gradient onto tangent space of sphere
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad


def _project_to_sphere(pts):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_init(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, max_iter=2000, ftol=1e-14, gtol=1e-12):
    """Run L-BFGS-B optimization on sphere-constrained points."""
    def objective_and_grad(flat):
        p = _project_to_sphere(flat.reshape(N, 3))
        e = _log_energy(p)
        g = _log_energy_grad(p)
        # Unproject gradient back to flat space for L-BFGS-B
        # The constraint is handled by re-projecting; gradient in flat space
        # is the projected gradient (since we always project)
        return e, g.flatten()
    
    x0 = pts.flatten()
    result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                               'maxls': 50, 'maxfun': 5000})
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    return best_pts, _log_energy(best_pts)


def _gradient_descent(pts, lr=0.01, n_iter=5000):
    """Simple gradient descent on sphere with adaptive step size."""
    current = pts.copy()
    best = current.copy()
    best_e = _log_energy(current)
    
    for it in range(n_iter):
        e = _log_energy(current)
        g = _log_energy_grad(current)
        
        # Adaptive learning rate
        norm_g = np.linalg.norm(g)
        if norm_g < 1e-14:
            break
        
        # Try step
        new_pts = _project_to_sphere(current - lr * g)
        new_e = _log_energy(new_pts)
        
        if new_e < e:
            current = new_pts
            if new_e < best_e:
                best = current.copy()
                best_e = new_e
            # Decrease lr slightly for stability
            lr *= 0.999
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return best, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Best result tracking
    best_pts = _fibonacci_spiral(N)
    best_energy = _log_energy(best_pts)
    
    # Strategy 1: L-BFGS from Fibonacci spiral
    def time_left():
        return deadline - time.time()
    
    if time_left() > 20:
        pts, e = _lbfgs_optimize(best_pts, max_iter=3000, ftol=1e-14, gtol=1e-12)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Strategy 2: Polish with tighter tolerances
    if time_left() > 15:
        pts, e = _lbfgs_optimize(best_pts, max_iter=2000, ftol=1e-15, gtol=1e-13)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Strategy 3: Try multiple random restarts with L-BFGS
    seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
    for seed in seeds:
        if time_left() < 15:
            break
        init_pts = _random_init(N, seed)
        pts, e = _lbfgs_optimize(init_pts, max_iter=1500, ftol=1e-13, gtol=1e-11)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Strategy 4: Try shifted Fibonacci spirals
    for offset in [0.3, 0.7, 0.1, 0.9]:
        if time_left() < 15:
            break
        init_pts = _fibonacci_spiral(N, offset)
        pts, e = _lbfgs_optimize(init_pts, max_iter=1500, ftol=1e-13, gtol=1e-11)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Strategy 5: Final polish pass with very tight tolerances
    if time_left() > 10:
        pts, e = _lbfgs_optimize(best_pts, max_iter=3000, ftol=1e-15, gtol=1e-14)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Strategy 6: Gradient descent polish if time remains
    if time_left() > 5:
        pts, e = _gradient_descent(best_pts, lr=0.001, n_iter=2000)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for validity
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
