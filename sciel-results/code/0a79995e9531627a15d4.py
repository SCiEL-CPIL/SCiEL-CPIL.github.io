"""Minimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100


def _energy_and_grad(pts):
    """Compute energy and gradient of log-energy on the sphere.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(d_ij)
    
    Gradient of E w.r.t. x_i (in R^3):
    dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    """
    # Compute pairwise squared distances efficiently
    # |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2*x_i.x_j
    # For unit vectors: = 2 - 2*x_i.x_j
    gram = pts @ pts.T  # (N, N)
    dist_sq = 2.0 - 2.0 * gram
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # Energy: E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    iu = np.triu_indices(N, k=1)
    d_sq_upper = dist_sq[iu]
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    # Gradient: grad[i] = sum_j (x_i - x_j) / d_ij^2
    # = x_i * sum_j 1/d_ij^2 - sum_j x_j * 1/d_ij^2
    inv_d_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_d_sq, 0.0)
    
    sum_inv = np.sum(inv_d_sq, axis=1)  # (N,)
    weighted_sum = inv_d_sq @ pts  # (N, 3): row i = sum_j inv_d_sq[i,j] * x_j
    
    grad = pts * sum_inv[:, None] - weighted_sum
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _golden_ratio_sphere(n):
    """Generate n points using golden ratio spiral (different phase from Fibonacci)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    phi = 2.0 * np.pi * (i / golden)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _jittered_fibonacci(n, seed=None, jitter_scale=0.1):
    """Fibonacci spiral with random jitter to escape local minima."""
    rng = np.random.RandomState(seed)
    base = _fibonacci_sphere(n)
    noise = rng.randn(n, 3) * jitter_scale
    pts = base + noise
    return _project_to_sphere(pts)


def _structured_random(n, seed=None, n_clusters=10):
    """Structured random: cluster points then randomize within clusters."""
    rng = np.random.RandomState(seed)
    # Create cluster centers on sphere
    centers = _random_sphere(n_clusters, seed=seed + 1000)
    # Assign each point to a cluster with random offset
    cluster_idx = rng.randint(0, n_clusters, n)
    pts = centers[cluster_idx]
    # Add random perturbation
    noise = rng.randn(n, 3) * 0.5
    pts = pts + noise
    return _project_to_sphere(pts)


def _optimize_from_start(start_pts, deadline, max_iter=3000):
    """Run L-BFGS optimization on the sphere, respecting deadline."""
    x0 = start_pts.flatten().copy()
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space of sphere
        # tangent gradient = g - (g . x) * x
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_tangent.flatten()
    
    try:
        res = minimize(
            objective, x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'maxfun': 80000,
                'ftol': 1e-15,
                'gtol': 1e-11,
                'maxls': 80
            }
        )
        result_pts = _project_to_sphere(res.x.reshape(N, 3))
        return result_pts
    except Exception:
        return None


def _polish(pts, deadline, max_iter=5000):
    """High-precision polish pass with very tight tolerances."""
    x0 = pts.flatten().copy()
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_tangent.flatten()
    
    try:
        res = minimize(
            objective, x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'maxfun': 200000,
                'ftol': 1e-16,
                'gtol': 1e-12,
                'maxls': 100
            }
        )
        result_pts = _project_to_sphere(res.x.reshape(N, 3))
        return result_pts
    except Exception:
        return pts


def _shake_and_reoptimize(pts, deadline, n_shakes=10):
    """Shake points with small random perturbations and re-optimize to escape local minima."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    for i in range(n_shakes):
        if time.time() > deadline - 3:
            break
        
        # Decreasing shake amplitude
        shake_scale = 0.02 * (1.0 - i / n_shakes) + 0.005
        noise = np.random.randn(N, 3) * shake_scale
        new_pts = _project_to_sphere(best_pts + noise)
        
        if not np.all(np.isfinite(new_pts)):
            continue
        
        # Quick optimization from shaken position
        result = _optimize_from_start(new_pts, deadline - 2, max_iter=500)
        if result is not None and np.all(np.isfinite(result)):
            e_new, _ = _energy_and_grad(result)
            if e_new < best_energy:
                best_energy = e_new
                best_pts = result
    
    return best_pts


def _local_refine(pts, deadline, n_iters=50):
    """Local refinement with small random perturbations and re-optimization."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    for _ in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        # Small random perturbation
        noise_scale = 0.05
        noise = np.random.randn(N, 3) * noise_scale
        new_pts = _project_to_sphere(best_pts + noise)
        
        if not np.all(np.isfinite(new_pts)):
            continue
        
        # Quick energy check
        e_new, _ = _energy_and_grad(new_pts)
        if e_new < best_energy:
            best_energy = e_new
            best_pts = new_pts
    
    return best_pts


def construct():
    """Main entry point: generate candidates, optimize, keep best."""
    start_time = time.time()
    # Reserve 15 seconds for safety before 150s limit
    deadline = start_time + 135.0
    
    # Best solution tracking
    best_pts = None
    best_energy = float('inf')
    
    np.random.seed(42)
    
    # Strategy 1: Multiple structured initializations
    # Fibonacci spiral
    if time.time() < deadline - 30:
        fib_pts = _fibonacci_sphere(N)
        result = _optimize_from_start(fib_pts, deadline - 10)
        if result is not None and np.all(np.isfinite(result)):
            e, _ = _energy_and_grad(result)
            if e < best_energy:
                best_energy = e
                best_pts = result
    
    # Golden ratio spiral
    if time.time() < deadline - 25:
        golden_pts = _golden_ratio_sphere(N)
        result = _optimize_from_start(golden_pts, deadline - 10)
        if result is not None and np.all(np.isfinite(result)):
            e, _ = _energy_and_grad(result)
            if e < best_energy:
                best_energy = e
                best_pts = result
    
    # Jittered Fibonacci
    if time.time() < deadline - 25:
        jitter_pts = _jittered_fibonacci(N, seed=42, jitter_scale=0.1)
        result = _optimize_from_start(jitter_pts, deadline - 10)
        if result is not None and np.all(np.isfinite(result)):
            e, _ = _energy_and_grad(result)
            if e < best_energy:
                best_energy = e
                best_pts = result
    
    # Strategy 2: Multiple diverse random restarts with L-BFGS
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        seed = 100 + i
        if i % 3 == 0:
            # Pure random
            random_pts = _random_sphere(N, seed=seed)
        elif i % 3 == 1:
            # Structured random (clustered)
            random_pts = _structured_random(N, seed=seed, n_clusters=8 + i % 5)
        else:
            # Jittered Fibonacci with different seeds
            random_pts = _jittered_fibonacci(N, seed=seed, jitter_scale=0.05 + 0.02 * (i % 3))
        
        result = _optimize_from_start(random_pts, deadline - 10, max_iter=1500)
        if result is not None and np.all(np.isfinite(result)):
            e, _ = _energy_and_grad(result)
            if e < best_energy:
                best_energy = e
                best_pts = result
    
    # Strategy 3: Local refinement on the best solution
    if best_pts is not None and time.time() < deadline - 20:
        refined = _local_refine(best_pts, deadline - 15, n_iters=30)
        e_ref, _ = _energy_and_grad(refined)
        if e_ref < best_energy:
            best_energy = e_ref
            best_pts = refined
    
    # Strategy 4: Shake and re-optimize to escape shallow local minima
    if best_pts is not None and time.time() < deadline - 15:
        shaken = _shake_and_reoptimize(best_pts, deadline - 10, n_shakes=8)
        e_shake, _ = _energy_and_grad(shaken)
        if e_shake < best_energy:
            best_energy = e_shake
            best_pts = shaken
    
    # Strategy 5: High-precision polish pass with maximum effort
    if best_pts is not None and time.time() < deadline - 5:
        # First polish with moderate settings
        polished1 = _polish(best_pts, deadline - 3, max_iter=3000)
        e_pol1, _ = _energy_and_grad(polished1)
        if e_pol1 < best_energy:
            best_energy = e_pol1
            best_pts = polished1
        
        # Second polish with tighter settings if time permits
        if time.time() < deadline - 2:
            polished2 = _polish(best_pts, deadline - 1, max_iter=5000)
            e_pol2, _ = _energy_and_grad(polished2)
            if e_pol2 < best_energy:
                best_energy = e_pol2
                best_pts = polished2
    
    # Final validation
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any issues
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    # Verify minimum distance
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
