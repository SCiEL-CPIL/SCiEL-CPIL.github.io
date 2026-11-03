"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # pts is (N, 3) array of unit vectors
    # Use pairwise distances
    # To avoid self-distances and double counting, use upper triangle
    from numpy.linalg import norm
    # Efficient: compute all pairwise distances
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    # dists = np.linalg.norm(diff, axis=2)  # (N, N)
    # Use upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = pts[i_idx] - pts[j_idx]
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    # Avoid log(0)
    dists = np.maximum(dists, 1e-15)
    # Energy is sum of log(1/d) = -sum(log(d))
    # Use fsum for precision
    terms = -np.log(dists)
    return math.fsum(terms)


def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points on the sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    d_ij = |x_i - x_j|
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-30:
                continue
            grad[i] += diff / dist_sq
    return grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _golden_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_on_sphere(rng):
    """Generate a random point on the unit sphere."""
    v = rng.standard_normal(3)
    norm = np.linalg.norm(v)
    return v / norm


def _local_optimize(pts, max_iter=2000, tol=1e-12, time_budget=30.0):
    """Local optimization using gradient descent with projection to sphere."""
    pts = _project_to_sphere(pts.copy())
    start_time = time.time()
    
    # Adaptive step size
    step = 0.01
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        grad = _log_energy_grad(pts)
        
        # Project gradient to tangent space: g - (g·x)x
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Take a step
        new_pts = pts - step * grad_tangent
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
            # Increase step slightly if improving
            step = min(step * 1.05, 0.1)
        else:
            # Decrease step
            step *= 0.5
            if step < 1e-10:
                break
    
    return best_pts, best_energy


def _simulated_annealing(pts, time_budget=60.0):
    """Simulated annealing to escape local minima."""
    pts = _project_to_sphere(pts.copy())
    start_time = time.time()
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    current_energy = best_energy
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_budget and T > T_min:
        # Propose a move: perturb one or a few points
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            # Random perturbation on the sphere
            delta = rng.standard_normal(3) * 0.1
            new_pts[idx] = pts[idx] + delta
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_energy = _log_energy(new_pts)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with golden spiral
    pts = _golden_spiral()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Phase 1: Local optimization from golden spiral
    if time.time() < deadline - 50.0:
        local_pts, local_energy = _local_optimize(pts, time_budget=40.0)
        if local_energy < best_energy:
            best_energy = local_energy
            best_pts = local_pts
    
    # Phase 2: Multiple random restarts with local optimization
    rng = np.random.default_rng(123)
    n_restarts = 0
    while time.time() < deadline - 30.0:
        # Random initial configuration
        random_pts = np.array([_random_on_sphere(rng) for _ in range(N)])
        
        # Quick local optimization with smaller budget
        remaining = deadline - 30.0 - time.time()
        if remaining < 2.0:
            break
        local_pts, local_energy = _local_optimize(random_pts, max_iter=500, time_budget=min(remaining * 0.5, 15.0))
        
        if local_energy < best_energy:
            best_energy = local_energy
            best_pts = local_pts
        
        n_restarts += 1
    
    # Phase 3: Simulated annealing from the best point found
    if time.time() < deadline - 10.0:
        sa_pts, sa_energy = _simulated_annealing(best_pts, time_budget=min(deadline - 10.0 - time.time(), 20.0))
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_pts
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2.0:
        remaining = deadline - 2.0 - time.time()
        if remaining > 1.0:
            final_pts, final_energy = _local_optimize(best_pts, max_iter=5000, tol=1e-14, time_budget=remaining)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
    
    # Ensure output is valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = best_pts[i_idx] - best_pts[j_idx]
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    if np.any(dists < 1e-9):
        # Fallback to golden spiral if points are too close
        best_pts = _golden_spiral()
    
    return best_pts

# EVOLVE-BLOCK-END
