"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 150
DEADLINE_SECONDS = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient with respect to pts on the sphere.
    
    For points on unit sphere, the gradient of log(1/|x_i - x_j|) wrt x_i is:
        (x_i - x_j) / |x_i - x_j|^2  (projected to tangent space)
    
    But since we optimize in R^3 with projection, we compute the full gradient
    and project onto the tangent plane.
    
    Actually, for the constrained problem on the sphere, the gradient of the 
    objective restricted to the manifold is the projection of the Euclidean 
    gradient onto the tangent space at each point.
    
    d/dx_i [log(1/|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    
    So total gradient at x_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum over i<j of log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # factor[i,j] = 1 / dist[i,j]^2
    factor = 1.0 / (dist * dist)  # (n, n)
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i,j] * (x_i - x_j) = sum_j factor[i,j] * diff[i,j]
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)  # (n, 3)
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x_i) * x_i
    # Since |x_i| = 1, projection is grad - (grad @ x_i) * x_i
    dot = np.sum(grad * pts, axis=1)  # (n,)
    grad_tangent = grad - dot[:, np.newaxis] * pts
    
    return energy, grad_tangent


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    def elapsed():
        return time.time() - start_time
    
    def has_time(seconds_needed):
        return elapsed() < DEADLINE_SECONDS - seconds_needed
    
    # Phase 1: Multi-start local optimization with different initializations
    # Try Fibonacci spiral and several random starts
    rng = np.random.default_rng(42)
    
    initializations = [best_pts.copy()]
    for _ in range(5):
        if not has_time(5.0):
            break
        initializations.append(_random_sphere(N, rng))
    
    # For each initialization, run L-BFGS-B with projection
    for init_idx, init_pts in enumerate(initializations):
        if not has_time(8.0):
            break
        
        # Project to sphere
        pts = _project_to_sphere(init_pts.copy())
        
        # Use L-BFGS-B on flattened array, with projection after each step
        # Actually, let's use a custom approach: optimize in R^3 and project
        # Better: use the fact that we can do gradient descent on the sphere
        
        # Simple gradient descent with backtracking line search on the sphere
        x = pts.flatten().copy()
        
        def objective(x):
            pts_local = x.reshape(N, 3)
            pts_local = _project_to_sphere(pts_local)
            e, _ = _log_energy_and_grad(pts_local)
            return e
        
        def gradient(x):
            pts_local = x.reshape(N, 3)
            pts_local = _project_to_sphere(pts_local)
            _, g = _log_energy_and_grad(pts_local)
            return g.flatten()
        
        # Run L-BFGS-B
        try:
            res = minimize(
                objective,
                x,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.success or res.fun < best_energy:
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e, _ = _log_energy_and_grad(candidate)
                if e < best_energy:
                    best_energy = e
                    best_pts = candidate
        except Exception:
            pass
        
        if not has_time(3.0):
            break
    
    # Phase 2: High-precision polish with more iterations
    if has_time(10.0):
        x = best_pts.flatten().copy()
        
        def objective(x):
            pts_local = x.reshape(N, 3)
            pts_local = _project_to_sphere(pts_local)
            e, _ = _log_energy_and_grad(pts_local)
            return e
        
        def gradient(x):
            pts_local = x.reshape(N, 3)
            pts_local = _project_to_sphere(pts_local)
            _, g = _log_energy_and_grad(pts_local)
            return g.flatten()
        
        try:
            res = minimize(
                objective,
                x,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.success or res.fun < best_energy:
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e, _ = _log_energy_and_grad(candidate)
                if e < best_energy:
                    best_energy = e
                    best_pts = candidate
        except Exception:
            pass
    
    # Phase 3: Final Newton-like polish using exact Hessian approximation
    # Do a few steps of conjugate gradient or Newton on the tangent space
    if has_time(5.0):
        pts = best_pts.copy()
        for _ in range(50):
            if not has_time(0.5):
                break
            e, g = _log_energy_and_grad(pts)
            
            # Check convergence
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                break
            
            # Simple gradient step with adaptive step size
            # Use line search
            alpha = 0.01
            for _ls in range(30):
                # Move in direction of negative gradient, project to sphere
                new_pts = pts - alpha * g
                new_pts = _project_to_sphere(new_pts)
                new_e, _ = _log_energy_and_grad(new_pts)
                
                if new_e < e:
                    pts = new_pts
                    e = new_e
                    break
                alpha *= 0.5
            
            # Update best
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
    
    # Final validation and return
    result = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Nudge slightly
                result[j] += 1e-8 * np.array([1, 0, 0])
                result[j] = result[j] / np.linalg.norm(result[j])
    
    return result

# EVOLVE-BLOCK-END
