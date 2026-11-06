"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

DEADLINE_SECONDS = 110.0  # Safety margin before 120s limit


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient with respect to x_i (constrained to sphere): 
      grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    (This is the gradient of the unconstrained energy; we project onto tangent space.)
    """
    n = len(pts)
    # Pairwise distances: d_ij = |x_i - x_j|
    # Use broadcasting: diff[i,j] = x_i - x_j, shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)  # (n, n)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum over i<j of log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, dist2[i,j] = |x_i - x_j|^2
    # grad[i] = sum_j diff[i,j] / dist2[i,j]  (j != i)
    inv_dist2 = 1.0 / dist2  # (n, n)
    np.fill_diagonal(inv_dist2, 0.0)
    # grad[i] = sum_j diff[i,j] * inv_dist2[i,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate Fibonacci sphere points with a rotation offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    golden = np.pi * (1.0 + np.sqrt(5.0))
    phi = golden * i + seed_offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-15)


def _momentum_descent(pts, max_iters=5000, lr_init=0.01, momentum=0.9, 
                      min_lr=1e-8, tol=1e-12, deadline=None):
    """Momentum-based gradient descent on the sphere with adaptive step size."""
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    energy, grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space: grad_t = grad - (grad . x) * x
    grad_t = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    vel = np.zeros_like(pts)
    lr = lr_init
    best_energy = energy
    best_pts = pts.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update velocity
        vel = momentum * vel - lr * grad_t
        # Project velocity to tangent space
        vel = vel - np.sum(vel * pts, axis=1, keepdims=True) * pts
        
        # Step
        new_pts = pts + vel
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, new_grad = _log_energy_and_grad(new_pts)
        
        if new_energy < energy:
            # Accept step
            pts = new_pts
            energy = new_energy
            grad = new_grad
            
            if energy < best_energy - 1e-10:
                best_energy = energy
                best_pts = pts.copy()
            
            # Decay learning rate slowly
            lr = max(min_lr, lr * 0.999)
        else:
            # Reject step, reduce learning rate
            lr = max(min_lr, lr * 0.5)
            vel *= 0.5
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < tol:
            break
    
    return best_pts, best_energy


def _anneal_restart(pts, temperature, max_iters=2000, deadline=None):
    """Perturb points with random noise and run descent."""
    n = len(pts)
    # Add random perturbation
    noise = np.random.standard_normal((n, 3)) * temperature
    perturbed = _project_to_sphere(pts + noise)
    return _momentum_descent(perturbed, max_iters=max_iters, lr_init=0.005, 
                             momentum=0.9, deadline=deadline)


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_pts = None
    best_energy = np.inf
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Strategy 1: Fibonacci sphere with various offsets
    for offset in [0.0, 0.5, 1.0, 2.0, 3.0]:
        if time.time() > deadline:
            break
        pts0 = _fibonacci_sphere(N, seed_offset=offset)
        pts, energy = _momentum_descent(pts0, max_iters=3000, lr_init=0.01, 
                                        momentum=0.9, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Strategy 2: Random initializations
    for _ in range(10):
        if time.time() > deadline:
            break
        pts0 = _random_sphere(N, rng)
        pts, energy = _momentum_descent(pts0, max_iters=2000, lr_init=0.005, 
                                        momentum=0.9, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Strategy 3: Annealing from best solution
    if best_pts is not None:
        for temp in [0.1, 0.05, 0.02, 0.01]:
            if time.time() > deadline:
                break
            pts, energy = _anneal_restart(best_pts, temperature, max_iters=1500, 
                                          deadline=deadline)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
    
    # Final refinement with many iterations
    if best_pts is not None:
        if time.time() < deadline - 5.0:
            pts, energy = _momentum_descent(best_pts, max_iters=5000, lr_init=0.001, 
                                            momentum=0.95, deadline=deadline)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
    
    # Ensure validity
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
