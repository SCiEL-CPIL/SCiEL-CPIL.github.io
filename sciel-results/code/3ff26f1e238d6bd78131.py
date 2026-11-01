"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy_and_grad(X):
    """Compute logarithmic energy and its Euclidean gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i (Euclidean) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    N = X.shape[0]
    # Pairwise distances
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / dist_ij^2
    # 1/dist^2 matrix
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff is (N, N, 3), inv_dist_sq is (N, N)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (N, 3)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere(n):
    """Generate n points on the sphere using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _riemannian_descent(X, max_iters=5000, tol=1e-12, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere.
    
    Uses the Riemannian gradient (tangential projection of Euclidean gradient)
    and exponential map (rotation) for updates.
    """
    X = _project_to_sphere(X.copy())
    N = X.shape[0]
    
    # Precompute for efficiency
    best_energy, _ = _log_energy_and_grad(X)
    best_X = X.copy()
    
    lr = 0.01  # initial learning rate
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, euclidean_grad = _log_energy_and_grad(X)
        
        # Riemannian gradient: project Euclidean gradient to tangent space
        # grad_tangent = grad - <grad, x> * x
        dot = np.sum(euclidean_grad * X, axis=1, keepdims=True)
        rgrad = euclidean_grad - dot * X
        
        # Check convergence
        grad_norm = np.linalg.norm(rgrad)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = lr
        for backtrack in range(30):
            # Exponential map: rotate each point by alpha * rgrad_i / |rgrad_i|
            # Simpler: use retraction X_new = normalize(X + alpha * rgrad)
            X_trial = X + alpha * rgrad
            X_trial = _project_to_sphere(X_trial)
            
            trial_energy, _ = _log_energy_and_grad(X_trial)
            
            if trial_energy < energy:
                break
            alpha *= 0.5
        
        # Update
        X = X_trial
        energy, _ = _log_energy_and_grad(X)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Adaptive learning rate
        if trial_energy < energy - 1e-10:
            lr = min(lr * 1.1, 0.1)
        else:
            lr = max(lr * 0.9, 1e-6)
    
    return best_X, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Multiple starting points
    starts = [_fibonacci_sphere(N)]
    
    # Add some random starts
    n_random_starts = 10
    for _ in range(n_random_starts):
        starts.append(_random_sphere(N, rng))
    
    best_X = None
    best_energy = float('inf')
    
    for start in starts:
        if time.time() > deadline - 5.0:
            break
        
        X_opt, energy_opt = _riemannian_descent(start, max_iters=3000, tol=1e-12, deadline=deadline)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_X = X_opt.copy()
    
    # If we have time, do a final polish on the best result
    if best_X is not None and time.time() < deadline - 2.0:
        X_polished, energy_polished = _riemannian_descent(
            best_X, max_iters=5000, tol=1e-14, deadline=deadline
        )
        if energy_polished < best_energy:
            best_X = X_polished
            best_energy = energy_polished
    
    # Final validation
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    # Ensure points are on the unit sphere
    best_X = _project_to_sphere(best_X)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Perturb
                best_X[j] += rng.standard_normal(3) * 1e-6
                best_X[j] /= np.linalg.norm(best_X[j])
    
    return best_X
# EVOLVE-BLOCK-END
