"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the unit sphere.
    
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where grad is the gradient on the sphere (projected).
    """
    # Pairwise distances
    # d_ij = |x_i - x_j|
    # energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # grad_i = sum_{j != i} (x_i - x_j) / (d_ij^2)  ... wait let me re-derive
    
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i for a given pair (i,j): -d/dx_i log(|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # But we need to project onto the sphere tangent space: grad_i - (grad_i . x_i) x_i
    
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute all pairwise differences
    # diff[i,j] = x_i - x_j, shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    
    # Distances
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum over i<j of log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    energy = -np.sum(np.log(dist_upper))
    
    # Gradient: for each i, grad_i = -sum_{j != i} (x_i - x_j) / dist_ij^2
    # Note: diff[i,j] = x_i - x_j
    # So grad_i = -sum_j (diff[i,j] / dist_sq[i,j])
    # But we need to be careful: this is the Euclidean gradient, then project to sphere
    
    # grad_unproj[i] = -sum_{j != i} diff[i,j] / dist_sq[i,j]
    # = -sum_{j != i} (x_i - x_j) / dist_sq[i,j]
    
    # For efficiency, compute using matrix operations
    # weight[i,j] = 1/dist_sq[i,j] for i != j, 0 for i == j
    weight = np.zeros_like(dist_sq)
    weight[iu[0], iu[1]] = 1.0 / dist_sq[iu[0], iu[1]]
    weight[iu[1], iu[0]] = 1.0 / dist_sq[iu[1], iu[0]]
    
    # grad_unproj[i] = -sum_j weight[i,j] * diff[i,j]
    # diff[i,j] = x_i - x_j
    # sum_j weight[i,j] * diff[i,j] = sum_j weight[i,j] * x_i - sum_j weight[i,j] * x_j
    # = x_i * sum_j weight[i,j] - sum_j weight[i,j] * x_j
    
    w_sum = np.sum(weight, axis=1)  # (n,)
    w_x = weight @ pts  # (n, 3) = sum_j weight[i,j] * x_j
    
    grad_unproj = -(pts * w_sum[:, np.newaxis] - w_x)
    
    # Project onto tangent space: grad_sphere = grad_unproj - (grad_unproj . x_i) x_i
    dot = np.sum(grad_unproj * pts, axis=1)  # (n,)
    grad_sphere = grad_unproj - dot[:, np.newaxis] * pts
    
    return energy, grad_sphere


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate Fibonacci sphere points with optional offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(start_pts, time_budget, deadline):
    """Run L-BFGS-B optimization from a starting point, with time guard."""
    start_time = time.time()
    
    # Flatten to 1D for scipy
    x0 = start_pts.flatten().copy()
    
    def objective_and_grad(x):
        pts = x.reshape(-1, 3)
        # Normalize to handle any drift
        pts = _normalize(pts)
        energy, grad = _log_energy_and_grad(pts)
        # Unproject gradient back to full space (since we normalize, the gradient
        # in the full space is just the sphere gradient, but we need to account
        # for the normalization constraint)
        # Actually, since we normalize inside, the effective gradient in x-space
        # is: for each point, the gradient of E w.r.t. x_i after normalization.
        # If we normalize x_i to x_i/|x_i|, then dE/dx_i = grad_sphere_i / |x_i| 
        # (approximately, when |x_i| is close to 1)
        # More precisely: if y_i = x_i / |x_i|, then dE/dx_i = (I - y_i y_i^T) grad_E(y_i) / |x_i|
        # But since we're on the sphere, |x_i| = 1, so dE/dx_i = grad_sphere_i
        # However, scipy will move x off the sphere, so we need the full gradient.
        # Let's just use the sphere gradient; the normalization handles the constraint.
        return energy, grad.flatten()
    
    # Use L-BFGS-B with time limit
    best_energy = np.inf
    best_x = x0.copy()
    
    # We'll do a simple iterative approach with time checks
    # scipy's minimize doesn't easily support time limits, so we'll do manual iterations
    
    # Actually, let's use scipy.optimize.minimize with a callback that checks time
    stop_flag = [False]
    
    def callback(xk):
        if time.time() > deadline - 2.0:
            stop_flag[0] = True
        return stop_flag[0]
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'callback': callback,
                'disp': False,
            }
        )
        
        if result.success or result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Gradient descent polish with adaptive step size
    if time.time() < deadline - 5.0:
        pts = _normalize(best_x.reshape(-1, 3))
        lr = 0.01
        min_lr = 1e-10
        max_steps = 2000
        prev_energy = best_energy
        
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            
            energy, grad = _log_energy_and_grad(pts)
            
            if energy < best_energy:
                best_energy = energy
                best_x = pts.flatten().copy()
            
            # Adaptive step size
            pts_new = pts - lr * grad
            pts_new = _normalize(pts_new)
            
            energy_new, _ = _log_energy_and_grad(pts_new)
            
            if energy_new < energy:
                pts = pts_new
                lr = min(lr * 1.1, 0.1)
            else:
                lr = max(lr * 0.5, min_lr)
                if lr < min_lr:
                    break
            
            # Check convergence
            if abs(prev_energy - energy) < 1e-12:
                break
            prev_energy = energy
    
    return _normalize(best_x.reshape(-1, 3)), best_energy


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci sphere
    starts.append(_fibonacci_sphere(N))
    
    # 2. Fibonacci sphere with different offsets
    for offset in [0.1, 0.3, 0.7, 1.5]:
        starts.append(_fibonacci_sphere(N, seed_offset=offset))
    
    # 3. Random starts
    for _ in range(5):
        starts.append(_random_sphere(N, rng))
    
    # 4. Anisotropic scaling of Fibonacci sphere
    for scale in [1.2, 0.8, 1.5]:
        pts = _fibonacci_sphere(N).copy()
        pts[:, 0] *= scale
        starts.append(_normalize(pts))
    
    best_pts = starts[0]
    best_energy = np.inf
    
    for start_pts in starts:
        if time.time() > deadline - 10.0:
            break
        
        pts, energy = _optimize_from_start(start_pts, 0, deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Final polish on the best result
    if time.time() < deadline - 3.0:
        pts, energy = _optimize_from_start(best_pts, 0, deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Ensure all points are on the unit sphere
    best_pts = _normalize(best_pts)
    
    # Validate: check no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci sphere
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
