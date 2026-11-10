"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 135.0  # 15s safety margin for 150s limit


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))


def _energy_and_grad(points):
    """Compute energy and gradient on the sphere (tangent-projected)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.maximum(dists, 1e-15)
    
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = float(np.sum(np.log(1.0 / d)))
    
    return energy, grad_tangent


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, 1e-15)


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n, rng=None):
    """Generate n points using Halton sequence (low-discrepancy)."""
    if rng is None:
        rng = np.random.default_rng(123)
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        z = 2 * u - 1
        theta = 2 * np.pi * v
        r = np.sqrt(max(0, 1 - z * z))
        points.append([r * np.cos(theta), r * np.sin(theta), z])
    
    return np.array(points)


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _adam_optimize(points, lr=0.005, max_iters=3000, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer on tangent space with adaptive learning rate."""
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    t = 0
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    for it in range(max_iters):
        if _remaining_time() < 12.0:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = -lr * m_hat / (np.sqrt(v_hat) + eps)
        update_tangent = update - np.sum(update * points, axis=1, keepdims=True) * points
        
        points = points + update_tangent
        points = _project_to_sphere(points)
        
        # Adaptive learning rate
        if energy > best_energy + 5.0:
            lr *= 0.7
    
    return best_points, best_energy


def _lbfgs_optimize(points, max_iters=5000, tol=1e-13):
    """L-BFGS-B optimization with projection to sphere."""
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol,
                                   'maxls': 50})
        
        if result.fun < _energy(points):
            return _project_to_sphere(result.x.reshape(n, 3)), float(result.fun)
    except Exception:
        pass
    
    return points, _energy(points)


def _anneal(points, rng, n_steps=500, t_start=1.0, t_end=0.01):
    """Simple simulated annealing on the sphere."""
    n = len(points)
    points = _project_to_sphere(points.copy())
    best_points = points.copy()
    best_energy = _energy(points)
    
    for step in range(n_steps):
        if _remaining_time() < 12.0:
            break
        
        t = t_start * (t_end / t_start) ** (step / max(1, n_steps - 1))
        
        idx = rng.integers(0, n)
        direction = rng.standard_normal(3)
        direction = direction / np.linalg.norm(direction)
        tangent = direction - np.dot(direction, points[idx]) * points[idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-15:
            tangent = tangent / tangent_norm
        
        step_size = 0.1 * t
        new_point = points[idx] + step_size * tangent
        new_point = _project_to_sphere(new_point.reshape(1, 3))[0]
        
        old_energy = best_energy
        points[idx] = new_point
        new_energy = _energy(points)
        
        delta = new_energy - old_energy
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points.copy()
        else:
            points[idx] = best_points[idx]
    
    return best_points, best_energy


def _high_precision_polish(points, max_iters=10000, tol=1e-15):
    """Final high-precision polish with very tight tolerance."""
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol,
                                   'maxls': 100, 'iprint': -1})
        
        if result.fun < _energy(points):
            return _project_to_sphere(result.x.reshape(n, 3)), float(result.fun)
    except Exception:
        pass
    
    return points, _energy(points)


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # 2. Halton sequence
    initializations.append(_halton_sphere(N, rng))
    
    # 3. Random
    initializations.append(_random_sphere(N, rng))
    
    # 4. Perturbed Fibonacci
    noise = rng.standard_normal((N, 3)) * 0.1
    initializations.append(_project_to_sphere(_fibonacci_sphere(N) + noise))
    
    # 5. Another random with different seed
    rng2 = np.random.default_rng(99)
    initializations.append(_random_sphere(N, rng2))
    
    # Optimize each initialization
    best_points = None
    best_energy = float('inf')
    
    for idx, init in enumerate(initializations):
        if _remaining_time() < 25.0:
            break
        
        # First pass: Adam for fast convergence
        pts, energy = _adam_optimize(init, lr=0.005, max_iters=2000)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
        
        # Second pass: L-BFGS for fine-tuning
        if _remaining_time() > 20.0:
            pts2, energy2 = _lbfgs_optimize(pts, max_iters=3000, tol=1e-13)
            if energy2 < best_energy:
                best_energy = energy2
                best_points = pts2.copy()
    
    # If we have time, do a final polish on the best result
    if _remaining_time() > 15.0 and best_points is not None:
        pts_final, energy_final = _lbfgs_optimize(best_points, max_iters=5000, tol=1e-14)
        if energy_final < best_energy:
            best_energy = energy_final
            best_points = pts_final.copy()
    
    # If still have time, try annealing from best
    if _remaining_time() > 12.0 and best_points is not None:
        rng3 = np.random.default_rng(777)
        pts_anneal, energy_anneal = _anneal(best_points, rng3, n_steps=300)
        if energy_anneal < best_energy:
            best_energy = energy_anneal
            best_points = pts_anneal.copy()
            # Quick L-BFGS polish after annealing
            if _remaining_time() > 8.0:
                pts_polish, energy_polish = _lbfgs_optimize(best_points, max_iters=2000, tol=1e-13)
                if energy_polish < best_energy:
                    best_energy = energy_polish
                    best_points = pts_polish.copy()
    
    # Final high-precision polish if time permits
    if _remaining_time() > 5.0 and best_points is not None:
        pts_hp, energy_hp = _high_precision_polish(best_points, max_iters=5000, tol=1e-15)
        if energy_hp < best_energy:
            best_energy = energy_hp
            best_points = pts_hp.copy()
    
    # Ensure validity
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    # Verify no points are too close
    dists = np.linalg.norm(best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = dists[iu].min()
    if min_dist < 1e-9:
        best_points = _fibonacci_sphere(N)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
