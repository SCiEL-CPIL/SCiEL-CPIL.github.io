"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150

def _energy_and_grad(X):
    """Compute logarithmic energy and Riemannian gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Riemannian Gradient w.r.t. x_i = (I - x_i x_i^T) * (sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2)
    """
    n = X.shape[0]
    # Compute all pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero on diagonal
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Euclidean Gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / dist[i,j]^2
    # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # grad_eucl[i] = - sum_j diff[i,j] / dist2[i,j]
    eucl_grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: grad_riem[i] = eucl_grad[i] - (eucl_grad[i] . X[i]) X[i]
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_init(n, num_dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        points[i - 1] = [halton(i, 2), halton(i, 3), halton(i, 5)]
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    # Add some jitter
    rng = np.random.RandomState(42)
    points += rng.normal(0, 0.01, size=points.shape)
    return _project_to_sphere(points)

def _riemannian_gd(X0, max_iter=1000, tol=1e-10, time_limit=10.0):
    """Riemannian Gradient Descent with momentum and backtracking line search."""
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    alpha = 0.5  # initial step size
    c = 1e-4     # Armijo parameter
    momentum = 0.0  # start without momentum, add later
    velocity = np.zeros_like(X)
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Add momentum after first 20 iterations
        if k >= 20:
            momentum = 0.3
        elif k >= 10:
            momentum = 0.15
        
        # Compute effective gradient with momentum
        effective_grad = grad + momentum * velocity
        
        # Backtracking line search
        alpha_trial = alpha
        step = alpha_trial * effective_grad
        X_new = X - step
        X_new = _project_to_sphere(X_new)
        
        energy_new, _ = _energy_and_grad(X_new)
        
        # Armijo condition: f(x_new) <= f(x) - c * alpha * ||grad||^2
        count = 0
        while energy_new > energy - c * alpha_trial * grad_norm**2:
            alpha_trial *= 0.5
            step = alpha_trial * effective_grad
            X_new = X - step
            X_new = _project_to_sphere(X_new)
            energy_new, _ = _energy_and_grad(X_new)
            count += 1
            if alpha_trial < 1e-15 or count > 30:
                break
        
        if alpha_trial < 1e-15:
            # If stuck, try smaller base step
            alpha = max(alpha * 0.5, 1e-10)
            continue
            
        # Update velocity for momentum
        velocity = momentum * velocity - alpha_trial * grad
        
        X = X_new
        energy = energy_new
        _, grad = _energy_and_grad(X)
        grad_norm = np.linalg.norm(grad)
        
        # Adapt step size based on success
        if count == 0:
            alpha = min(alpha * 1.2, 1.0)
        else:
            alpha = max(alpha * 0.8, 1e-10)
        
        if grad_norm < tol:
            break
            
    return X, energy

def _farthest_point_init(n, rng_seed=42):
    """Farthest-point sampling initialization for better diversity."""
    rng = np.random.RandomState(rng_seed)
    points = np.zeros((n, 3))
    
    # Start with a random point
    points[0] = rng.randn(3)
    points[0] /= np.linalg.norm(points[0])
    
    min_dists = np.full(n, np.inf)
    
    for i in range(1, n):
        # Compute distance from each candidate to all existing points
        # Use the point farthest from all existing points
        best_idx = 0
        best_min_dist = -1
        
        for _ in range(100):  # Try 100 random candidates
            candidate = rng.randn(3)
            candidate /= np.linalg.norm(candidate)
            
            # Compute minimum distance to existing points
            dists = np.linalg.norm(points[:i] - candidate, axis=1)
            min_dist = np.min(dists)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = candidate
        
        points[i] = best_idx
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_init(N))
    
    # 2. Halton sequence
    initializations.append(_halton_init(N))
    
    # 3. Farthest-point sampling for better initial spread
    initializations.append(_farthest_point_init(N, rng_seed=42))
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(123)
    rand_pts = rng.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts))
    
    # 5. Random with different seed
    rng2 = np.random.RandomState(456)
    rand_pts2 = rng2.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts2))
    
    # 6. Another random seed
    rng3 = np.random.RandomState(789)
    rand_pts3 = rng3.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts3))
    
    # 7. Fibonacci with slight perturbation
    fib_perturbed = _fibonacci_init(N).copy()
    rng4 = np.random.RandomState(321)
    fib_perturbed += rng4.normal(0, 0.05, size=fib_perturbed.shape)
    initializations.append(_project_to_sphere(fib_perturbed))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: Riemannian GD from multiple starts
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
            
        # Use a time limit for each run to ensure we finish
        time_budget = (deadline - 30.0 - time.time()) / max(1, len(initializations) - init_idx)
        time_budget = max(time_budget, 5.0)
        
        try:
            X_opt, e_opt = _riemannian_gd(X0, max_iter=500, tol=1e-8, time_limit=time_budget)
            
            # Polish with L-BFGS-B for better local convergence
            def objective(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, _ = _energy_and_grad(X)
                return e
            
            def gradient(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, g = _energy_and_grad(X)
                return g.flatten()
                
            # Only polish if we have time
            if time.time() < deadline - 20.0:
                result = minimize(
                    objective,
                    X_opt.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'm': 15}
                )
                X_polished = _project_to_sphere(result.x.reshape(N, 3))
                e_polished, _ = _energy_and_grad(X_polished)
                
                if e_polished < e_opt:
                    X_opt = X_polished
                    e_opt = e_polished
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
                
        except Exception:
            pass

    # Phase 2: Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        try:
            X0 = best_points.copy()
            
            def objective(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, _ = _energy_and_grad(X)
                return e
            
            def gradient(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, g = _energy_and_grad(X)
                return g.flatten()
                
            # First pass with tighter tolerances
            result = minimize(
                objective,
                X0.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'm': 20}
            )
            
            X_opt = _project_to_sphere(result.x.reshape(N, 3))
            e_opt, _ = _energy_and_grad(X_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
            
            # Second polish pass if we have time
            if time.time() < deadline - 3.0 and e_opt < best_energy + 1e-10:
                result2 = minimize(
                    objective,
                    best_points.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15, 'm': 20}
                )
                X_opt2 = _project_to_sphere(result2.x.reshape(N, 3))
                e_opt2, _ = _energy_and_grad(X_opt2)
                
                if e_opt2 < best_energy:
                    best_energy = e_opt2
                    best_points = X_opt2
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        # Fall back to Fibonacci initialization
        return _fibonacci_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
