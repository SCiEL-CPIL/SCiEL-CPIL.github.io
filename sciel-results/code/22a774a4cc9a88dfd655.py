"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dist = np.log(dist[iu])
    energy = -np.sum(log_dist)
    
    # Gradient: d/dx_i sum_{j != i} log(1/|x_i - x_j|) = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # For each pair (i,j), contribution to grad_i is (x_j - x_i)/d^2, to grad_j is (x_i - x_j)/d^2
    grad = np.zeros_like(pts)
    # dist^2
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)
    
    # For each point i, grad_i = sum_{j != i} (x_j - x_i) / dist_sq[i,j]
    # (x_j - x_i) = -diff[i,j]
    # So grad_i = -sum_{j != i} diff[i,j] / dist_sq[i,j]
    # diff[i,j] = pts[i] - pts[j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad = -diff @ inv_dist_sq (sum over j for each i)
    # diff shape (n, n, 3), inv_dist_sq shape (n, n)
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _icosahedron_base():
    """Generate icosahedron vertices on unit sphere."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    return _project_to_sphere(verts)


def _optimize_from_start(pts, deadline):
    """Optimize from a starting configuration using L-BFGS-B."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    def objective(x):
        pts_local = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(pts_local)
        return e
    
    def gradient(x):
        pts_local = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(pts_local)
        return g.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 10000,
                'disp': False
            }
        )
        if result.success or result.fun < best_energy:
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy, _ = _energy_and_grad(best_pts)
    except Exception:
        pass
    
    return best_pts, best_energy


def _refine(pts, deadline):
    """Do a few more refinement steps with tighter tolerances."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    def objective(x):
        pts_local = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(pts_local)
        return e
    
    def gradient(x):
        pts_local = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(pts_local)
        return g.flatten()
    
    x0 = best_pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 50000,
                'disp': False
            }
        )
        if result.fun < best_energy:
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy, _ = _energy_and_grad(best_pts)
    except Exception:
        pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate multiple starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Random starts with different seeds
    for seed in range(5):
        starts.append(_random_sphere(N, seed=seed))
    
    # 3. Icosahedron-based: start from icosahedron and add more points
    ico = _icosahedron_base()
    # Create a denser starting point by perturbing and adding
    rng = np.random.default_rng(42)
    ico_start = ico.copy()
    # Add more points around existing ones
    extra = _random_sphere(N - 12, seed=123)
    starts.append(np.vstack([ico_start, extra]))
    
    # 4. Perturbed Fibonacci
    rng2 = np.random.default_rng(99)
    fib = _fibonacci_spiral(N)
    perturbed = fib + rng2.standard_normal((N, 3)) * 0.1
    starts.append(_project_to_sphere(perturbed))
    
    # Optimize from each start, keeping the best
    best_pts = starts[0]
    best_energy = float('inf')
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        
        pts, energy = _optimize_from_start(start_pts, deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Final refinement pass on the best result
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _refine(best_pts, deadline)
    
    # Ensure points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    # Final verification
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
