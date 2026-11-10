"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import random
from scipy.optimize import minimize


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute the gradient of the logarithmic energy with respect to points."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            factor = 1.0 / (d * d)
            grad[i] -= diff * factor
            grad[j] += diff * factor
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _conjugate_gradient_sphere(points, max_iter=200, tol=1e-10):
    """Conjugate gradient method on the sphere manifold."""
    n = len(points)
    current = points.copy()
    g = _log_energy_grad(current)
    # Project gradient to tangent space
    g_tan = g - np.sum(g * current, axis=1, keepdims=True) * current
    
    # Check if already at a critical point
    if np.linalg.norm(g_tan) < tol:
        return current
    
    # Fletcher-Reeves conjugate gradient
    d = -g_tan.copy()
    for k in range(max_iter):
        # Line search: find step size alpha
        best_alpha = 0.01
        best_f = _log_energy(current)
        for alpha in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            trial = current + alpha * d
            trial = _project_to_sphere(trial)
            f_trial = _log_energy(trial)
            if f_trial < best_f:
                best_f = f_trial
                best_alpha = alpha
        
        new_points = _project_to_sphere(current + best_alpha * d)
        
        # Compute new gradient
        g_new = _log_energy_grad(new_points)
        g_new_tan = g_new - np.sum(g_new * new_points, axis=1, keepdims=True) * new_points
        
        if np.linalg.norm(g_new_tan) < tol:
            return new_points
        
        # Update direction (Fletcher-Reeves)
        beta = np.sum(g_new_tan * g_new_tan) / max(np.sum(g_tan * g_tan), 1e-15)
        d = -g_new_tan + beta * d
        
        current = new_points
        g_tan = g_new_tan
    
    return current


def _simulated_annealing(points, n_iter=500, t_start=1.0, t_end=0.01):
    """Simulated annealing on the sphere."""
    n = len(points)
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    for it in range(n_iter):
        # Cooling schedule
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Pick a random point and perturb it
        idx = random.randint(0, n - 1)
        # Generate random direction
        rand_dir = np.random.randn(3)
        rand_dir /= np.linalg.norm(rand_dir)
        # Move in tangent direction
        tangent = rand_dir - np.dot(rand_dir, current[idx]) * current[idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            continue
        tangent /= tangent_norm
        
        step_size = 0.1 * t
        trial = current.copy()
        trial[idx] += step_size * tangent
        trial[idx] = trial[idx] / np.linalg.norm(trial[idx])
        
        trial_energy = _log_energy(trial)
        delta_e = trial_energy - current_energy
        
        # Accept with probability exp(-delta_e / t)
        if delta_e < 0 or random.random() < math.exp(-delta_e / max(t, 1e-10)):
            current = trial
            current_energy = trial_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
    
    return best


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)


def construct():
    # Multiple initial configurations
    best_points = None
    best_energy = float('inf')
    
    # Configuration 1: Fibonacci spiral
    pts1 = _fibonacci_spiral()
    
    # Configuration 2: Random points
    random.seed(42)
    np.random.seed(42)
    pts2 = np.random.randn(N, 3)
    pts2 = _project_to_sphere(pts2)
    
    # Configuration 3: Another Fibonacci variant with offset
    i = np.arange(N) + 0.3
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts3 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts3 = _project_to_sphere(pts3)
    
    initial_configs = [pts1, pts2, pts3]
    
    for config in initial_configs:
        # Phase 1: Simulated annealing to escape local minima
        sa_result = _simulated_annealing(config, n_iter=300, t_start=0.5, t_end=0.01)
        
        # Phase 2: Conjugate gradient refinement
        cg_result = _conjugate_gradient_sphere(sa_result, max_iter=100, tol=1e-12)
        
        energy = _log_energy(cg_result)
        
        if energy < best_energy:
            best_energy = energy
            best_points = cg_result
    
    # Final verification
    if best_points is None:
        best_points = pts1
    
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(best_points)):
        return _fibonacci_spiral()
    
    # Check for very close points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        return _fibonacci_spiral()
    
    return best_points
# EVOLVE-BLOCK-END
