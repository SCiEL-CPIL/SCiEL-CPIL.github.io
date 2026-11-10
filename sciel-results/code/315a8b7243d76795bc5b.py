"""
Strategy: Basin Hopping with Adaptive Step Sizes.
Instead of a single MD trajectory or pure SA, we use a global optimization
method that repeatedly performs local optimization (L-BFGS) from perturbed
starting points, accepting worse minima with a temperature-based criterion.
This is structurally different from the current program's MD approach:
- Current: continuous trajectory with adaptive dt, no explicit local opt
- New: discrete jumps between local minima, explicit L-BFGS at each step
The perturbation magnitude and temperature are adapted based on acceptance rate.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_{ij})."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dist[mask]
    energy = -np.sum(np.log(valid_dists))
    return energy

def _energy_grad(pts):
    """Compute gradient of logarithmic energy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    # d/dx_i log(1/d_ij) = -d/dx_i log(d_ij) = -(x_i - x_j)/d_ij^2
    grad_terms = diff / dist_sq[..., np.newaxis]
    grad = np.sum(grad_terms, axis=1)
    return grad

def _project_tangent(pts, vec):
    """Project vector onto tangent space of sphere."""
    dot = np.sum(pts * vec, axis=1, keepdims=True)
    return vec - dot * pts

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    vecs = rng.standard_normal((n, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def _local_optimize(pts, max_iter=100, time_limit=None):
    """
    Perform local optimization using L-BFGS-B on the sphere.
    Returns (optimized_pts, energy, converged).
    """
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        p = p / norms
        e = _energy(p)
        g = _energy_grad(p)
        g_tan = _project_tangent(p, g)
        return e, g_tan.flatten()
    
    options = {
        'maxiter': max_iter,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'disp': False
    }
    
    if time_limit is not None:
        start = time.time()
        # Wrap with time check
        def timed_fun_and_grad(x):
            if time.time() - start > time_limit:
                raise TimeoutError
            return fun_and_grad(x)
        try:
            res = minimize(timed_fun_and_grad, x0, method='L-BFGS-B', jac=True, options=options)
        except TimeoutError:
            return pts, _energy(pts), False
    else:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, options=options)
    
    final_pts = res.x.reshape(-1, 3)
    norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    final_pts = final_pts / norms
    
    final_energy = _energy(final_pts)
    converged = res.success or res.status == 0
    
    return final_pts, final_energy, converged

def _perturb(pts, step_size, rng):
    """
    Perturb points by adding random tangent-space noise.
    step_size controls the magnitude of perturbation.
    """
    noise = rng.standard_normal((len(pts), 3)) * step_size
    noise = _project_tangent(pts, noise)
    new_pts = pts + noise
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_pts = new_pts / norms
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Initial state: Fibonacci sphere, then local optimize
    current_pts = _fibonacci_sphere(N)
    current_pts, current_energy, _ = _local_optimize(current_pts, max_iter=200)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    # Basin hopping parameters
    T = 1.0  # Temperature
    step_size = 0.1  # Initial perturbation size
    max_iter = 2000
    iter_count = 0
    
    # Track acceptance rate for adaptive step size
    accept_count = 0
    total_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Check time more frequently
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Perturb current point
        trial_pts = _perturb(current_pts, step_size, rng)
        
        # Local optimize from perturbed point
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        
        # Use less iterations when time is tight
        if time_left < 10.0:
            max_iter_local = 30
        elif time_left < 30.0:
            max_iter_local = 100
        else:
            max_iter_local = 200
        
        trial_pts, trial_energy, _ = _local_optimize(trial_pts, max_iter=max_iter_local, time_limit=max(1.0, time_left - 2.0))
        
        # Metropolis acceptance criterion
        delta_E = trial_energy - current_energy
        total_count += 1
        
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            # Accept
            accept_count += 1
            current_pts = trial_pts
            current_energy = trial_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        # Adaptive step size based on acceptance rate
        if total_count > 20:
            accept_rate = accept_count / total_count
            # Target acceptance rate ~0.5
            if accept_rate > 0.7:
                step_size *= 1.1
            elif accept_rate < 0.3:
                step_size *= 0.9
            step_size = min(step_size, 0.5)
            step_size = max(step_size, 0.01)
        
        # Cool down temperature
        T *= 0.995
        T = max(T, 0.01)
        
        # Periodic restart: if stuck, try a random restart
        if iter_count % 100 == 0 and accept_count / max(1, total_count) < 0.1:
            # Reset with a random start
            current_pts = _random_sphere(N, rng)
            current_pts, current_energy, _ = _local_optimize(current_pts, max_iter=100)
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
            # Reset counters
            accept_count = 0
            total_count = 0
            step_size = 0.1
            T = 1.0
    
    # Final polish on best point
    time_left = deadline - time.time()
    if time_left > 5.0:
        final_pts, final_energy, _ = _local_optimize(best_pts, max_iter=300, time_limit=max(2.0, time_left - 2.0))
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    # Validity check
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    # Ensure points are on unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    # Check for points too close together
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) <= 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
