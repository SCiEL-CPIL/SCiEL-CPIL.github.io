"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    X: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (this is the gradient of -log(|x_i-x_j|) w.r.t. x_i)
    """
    n = X.shape[0]
    # Compute pairwise distances
    # diff[i,j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    dist = np.sqrt(np.maximum(dist2, 1e-30))  # (n, n)
    
    # Avoid log(0) and division by zero on diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[triu_mask]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # This is the gradient of -log(|x_i - x_j|) w.r.t. x_i
    # grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    # Note: this equals sum_j (x_i - x_j) / dist2[i,j]
    
    # Set diagonal to 0 to avoid division by zero
    dist2_safe = dist2.copy()
    np.fill_diagonal(dist2_safe, 1.0)
    
    # grad[i] = sum_j (X[i] - X[j]) / dist2[i,j]
    # = sum_j X[i]/dist2[i,j] - sum_j X[j]/dist2[i,j]
    # = X[i] * sum_j 1/dist2[i,j] - sum_j X[j]/dist2[i,j]
    
    inv_dist2 = np.zeros_like(dist2_safe)
    inv_dist2[mask] = 1.0 / dist2_safe[mask]
    
    # sum_j 1/dist2[i,j] for each i
    sum_inv_dist2 = np.sum(inv_dist2, axis=1)  # (n,)
    
    # sum_j X[j] / dist2[i,j] for each i
    # This is inv_dist2 @ X
    sum_X_over_dist2 = inv_dist2 @ X  # (n, 3)
    
    grad = X * sum_inv_dist2[:, np.newaxis] - sum_X_over_dist2  # (n, 3)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 150 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_X = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_X)
    
    # Try multiple random restarts with L-BFGS-B style optimization
    # using projected gradient / steepest descent on the sphere
    
    rng = np.random.default_rng(42)
    
    # Strategy: use L-BFGS-B with sphere constraint handled by projection
    # Actually, let's use scipy.optimize.minimize with L-BFGS-B
    # but we need to handle the sphere constraint.
    
    # Better approach: use scipy's L-BFGS-B with a penalty or use
    # the fact that we can parameterize on the sphere.
    
    # Let's use a simple approach: L-BFGS-B with projection after each step
    # This is not strictly correct but often works.
    
    # Actually, let's use scipy.optimize.minimize with method='L-BFGS-B'
    # and handle the constraint by projecting back to sphere.
    
    from scipy.optimize import minimize
    
    def objective(params):
        """Objective function that takes flat parameters, projects to sphere, returns energy."""
        X = params.reshape(N, 3)
        X = _project_to_sphere(X)
        energy, _ = _energy_and_grad(X)
        return energy
    
    def gradient(params):
        """Gradient of objective w.r.t. flat parameters (projected gradient)."""
        X = params.reshape(N, 3)
        X = _project_to_sphere(X)
        energy, grad = _energy_and_grad(X)
        # Project gradient to tangent space: grad_tangent = grad - (grad . x) * x
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad_tangent = grad - dot * X
        return grad_tangent.flatten()
    
    def _optimize_from_start(X0, time_budget):
        """Run L-BFGS-B optimization from X0 with given time budget."""
        nonlocal best_X, best_energy
        nonlocal start_time, deadline
        
        X0 = _project_to_sphere(X0)
        
        # Define a time-limited callback
        def callback(params):
            if time.time() > time_budget:
                raise RuntimeError("time_limit")
            return None
        
        try:
            res = minimize(
                objective,
                X0.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxls': 50,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxfun': 1000,
                },
                callback=callback,
            )
            X_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt, _ = _energy_and_grad(X_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_X = X_opt
        except RuntimeError:
            pass
        except Exception:
            pass
        
        return best_X, best_energy
    
    # First, try from Fibonacci spiral
    _optimize_from_start(_fibonacci_spiral(N), deadline - 20)
    
    # Try several random restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Generate random starting points
        X_rand = rng.standard_normal((N, 3))
        X_rand = _project_to_sphere(X_rand)
        
        # Small perturbation of the best solution
        if r % 2 == 0:
            X_rand = best_X + 0.1 * rng.standard_normal((N, 3))
            X_rand = _project_to_sphere(X_rand)
        
        _optimize_from_start(X_rand, deadline - 20)
    
    # Final verification: ensure all points are on unit sphere and well-separated
    best_X = _project_to_sphere(best_X)
    
    # Check for points too close together
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    min_dist = np.sqrt(np.min(dist2))
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral(N)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_spiral(N)
    
    return best_X

# EVOLVE-BLOCK-END
