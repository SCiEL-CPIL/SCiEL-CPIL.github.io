"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 100

# EVOLVE-BLOCK-START

def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Pairwise distances
    # Use broadcasting for efficiency
    # dist_sq[i,j] = |x_i - x_j|^2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid self-distances
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    terms = np.log(1.0 / dist[iu])
    return math.fsum(terms.tolist())

def _energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # dE/dx_i = sum_{j != i} -1/(d_ij * |x_i - x_j|) * (x_i - x_j) / |x_i - x_j|
    # Actually: d/dx_i log(1/|x_i - x_j|) = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j| = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = sum_{j != i} -(x_i - x_j) / |x_i - x_j|^2
    #            = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    # inv_dist_sq has 1 on diagonal, we want 0 on diagonal
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    #         = (sum_j x_j * inv_dist_sq[i,j]) - x_i * (sum_j inv_dist_sq[i,j])
    #         = (inv_dist_sq @ x) - x * (inv_dist_sq @ ones)
    row_sums = inv_dist_sq.sum(axis=1)
    grad = inv_dist_sq @ points - points * row_sums[:, np.newaxis]
    return grad

def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_spiral(n):
    """Generate points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _lbfgs_optimize(points, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """Optimize points on sphere using L-BFGS with projection."""
    if start_time is None:
        start_time = time.time()
    
    def objective_and_grad(x_flat):
        if time_limit is not None and time.time() - start_time > time_limit:
            return None, None
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        e = _energy(x)
        g = _energy_grad(x)
        return e, g.flatten()
    
    x0 = points.flatten()
    result = minimize(
        lambda x: objective_and_grad(x)[0],
        x0,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if result is not None and result.x is not None:
        best_points = _project_to_sphere(result.x.reshape(-1, 3))
        return best_points, _energy(best_points)
    return points, _energy(points)

def _newton_step(points, max_iter=20, time_limit=None, start_time=None):
    """Newton-Raphson optimization on the sphere using Riemannian Hessian approximation."""
    if start_time is None:
        start_time = time.time()
    
    best_points = points.copy()
    best_energy = _energy(points)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        x = best_points
        e = _energy(x)
        g = _energy_grad(x)
        
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        # Approximate Hessian using finite differences of gradient
        # This is expensive but more accurate than L-BFGS for small N
        h = 1e-6
        n = len(x)
        # Use a simpler approach: use L-BFGS style with more iterations
        # Actually, let's use a trust-region-like approach
        
        # Project gradient to tangent space
        # The constraint is |x_i| = 1, so tangent space is orthogonal to x_i
        # For each point, the tangent gradient is g_i - (g_i · x_i) x_i
        tangent_g = g.copy()
        for i in range(n):
            tangent_g[i] -= np.dot(g[i], x[i]) * x[i]
        
        # Take a step in the negative tangent gradient direction
        # Use line search to determine step size
        alpha = 1.0
        for ls_iter in range(50):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            # Move in tangent direction and project back
            new_x = x - alpha * tangent_g
            new_x = _project_to_sphere(new_x)
            new_e = _energy(new_x)
            if new_e < e - 1e-4 * alpha * np.dot(tangent_g, g):
                best_points = new_x
                best_energy = new_e
                break
            alpha *= 0.5
        
        if alpha < 1e-10:
            break
    
    return best_points, best_energy

def _simulated_annealing(points, n_steps=5000, t_start=0.1, t_end=0.001, time_limit=None, start_time=None):
    """Simulated annealing to escape local minima."""
    if start_time is None:
        start_time = time.time()
    
    best_points = points.copy()
    best_energy = _energy(points)
    current_points = points.copy()
    current_energy = best_energy
    
    n = len(current_points)
    t = t_start
    
    for step in range(n_steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Cooldown
        t = t_start * (t_end / t_start) ** (step / max(1, n_steps - 1))
        
        # Generate random perturbation
        rng = np.random
        idx = rng.randint(n)
        angle = rng.uniform(0, 2 * np.pi)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        
        # Rotate point by small angle
        theta = rng.uniform(-0.1, 0.1)
        # Rodrigues rotation
        c = np.cos(theta)
        s = np.sin(theta)
        k = axis
        new_p = current_points[idx] * c + np.cross(k, current_points[idx]) * s + k * np.dot(k, current_points[idx]) * (1 - c)
        new_p = new_p / np.linalg.norm(new_p)
        
        new_points = current_points.copy()
        new_points[idx] = new_p
        new_energy = _energy(new_points)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
            current_points = new_points
            current_energy = new_energy
            if new_energy < best_energy:
                best_points = new_points.copy()
                best_energy = new_energy
    
    return best_points, best_energy

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s safety margin for 120s limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy(points)
    
    # Phase 1: L-BFGS optimization
    points, energy = _lbfgs_optimize(points, max_iter=100, time_limit=time_limit, start_time=start_time)
    if energy < best_energy:
        best_points = points.copy()
        best_energy = energy
    
    # Phase 2: Simulated annealing to escape local minima
    if time.time() - start_time < time_limit - 20:
        points, energy = _simulated_annealing(
            best_points, n_steps=3000, t_start=0.05, t_end=0.001,
            time_limit=time_limit, start_time=start_time
        )
        if energy < best_energy:
            best_points = points.copy()
            best_energy = energy
    
    # Phase 3: Another L-BFGS pass from the annealed result
    if time.time() - start_time < time_limit - 10:
        points, energy = _lbfgs_optimize(best_points, max_iter=200, time_limit=time_limit, start_time=start_time)
        if energy < best_energy:
            best_points = points.copy()
            best_energy = energy
    
    # Phase 4: Final fine-tuning with Newton-like steps
    if time.time() - start_time < time_limit - 5:
        points, energy = _newton_step(best_points, max_iter=10, time_limit=time_limit, start_time=start_time)
        if energy < best_energy:
            best_points = points.copy()
            best_energy = energy
    
    # Validate
    best_points = _project_to_sphere(best_points)
    
    # Check for close points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(dist_sq.min())
    if min_dist < 1e-9:
        # Perturb the closest pair
        iu = np.argmin(dist_sq)
        i, j = np.unravel_index(iu, dist_sq.shape)
        best_points[j] += 1e-6 * np.random.randn(3)
        best_points = _project_to_sphere(best_points)
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_spiral(N)
    
    return best_points

# EVOLVE-BLOCK-END
