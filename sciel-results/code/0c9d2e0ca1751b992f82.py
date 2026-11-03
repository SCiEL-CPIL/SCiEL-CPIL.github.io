"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _compute_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return -1e18
            total += math.log(1.0 / d)
    return total


def _compute_energy_fast(pts):
    """Vectorized energy computation."""
    n = len(pts)
    # Compute all pairwise distances
    # Using broadcasting: shape (n, n, 3) -> (n, n)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid log(0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    return np.sum(np.log(1.0 / dists[iu]))


def _compute_gradient(pts):
    """
    Compute gradient of the logarithmic energy w.r.t. points.
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                continue
            grad[i] += d / (dist * dist)
    return grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _gradient_descent(pts, max_iter=5000, lr_init=0.01, tol=1e-10):
    """
    Gradient descent on the sphere using retraction (projection).
    Uses adaptive step size.
    """
    pts = pts.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _compute_energy_fast(pts)
    
    lr = lr_init
    prev_energy = best_energy
    
    for it in range(max_iter):
        grad = _compute_gradient(pts)
        
        # Move in negative gradient direction
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _compute_energy_fast(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        if new_energy < prev_energy:
            # Accept step, maybe increase lr slightly
            pts = new_pts
            prev_energy = new_energy
            if it > 100:
                lr = min(lr * 1.05, 0.1)
        else:
            # Reject step, decrease lr
            lr *= 0.5
            if lr < 1e-8:
                break
        
        if abs(prev_energy - best_energy) < tol:
            break
    
    return best_pts, best_energy


def _anneal(pts, max_iter=3000, T_init=0.5, T_final=1e-6, lr=0.01):
    """
    Simple simulated annealing on the sphere.
    """
    pts = pts.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _compute_energy_fast(pts)
    current_energy = best_energy
    
    for it in range(max_iter):
        T = T_init * (T_final / T_init) ** (it / max_iter)
        
        # Random perturbation
        noise = np.random.randn(n, 3) * lr
        new_pts = pts + noise
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _compute_energy_fast(new_pts)
        
        delta = new_energy - current_energy
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            pts = new_pts
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
    
    return best_pts, best_energy


def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    Uses multiple strategies: Fibonacci start, gradient descent, annealing,
    and multi-restart.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial points from Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    
    # Strategy 1: Gradient descent from Fibonacci start
    best_pts, best_energy = _gradient_descent(pts0, max_iter=3000, lr_init=0.01)
    
    # Strategy 2: Annealing from Fibonacci start
    if time.time() < deadline:
        pts_a, energy_a = _anneal(pts0, max_iter=2000, T_init=0.3, T_final=1e-5, lr=0.05)
        if energy_a < best_energy:
            best_energy = energy_a
            best_pts = pts_a
    
    # Strategy 3: Gradient descent from annealed result
    if time.time() < deadline:
        pts_gd2, energy_gd2 = _gradient_descent(best_pts, max_iter=3000, lr_init=0.005)
        if energy_gd2 < best_energy:
            best_energy = energy_gd2
            best_pts = pts_gd2
    
    # Strategy 4: Multi-restart with random perturbations
    for restart in range(10):
        if time.time() >= deadline:
            break
        
        # Random perturbation of the best solution
        noise = np.random.randn(N, 3) * 0.1
        pts_r = best_pts + noise
        pts_r = _project_to_sphere(pts_r)
        
        # Gradient descent
        pts_opt, energy_opt = _gradient_descent(pts_r, max_iter=2000, lr_init=0.01)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
        
        # Anneal from this
        if time.time() < deadline:
            pts_an, energy_an = _anneal(pts_opt, max_iter=1000, T_init=0.1, T_final=1e-5, lr=0.03)
            if energy_an < best_energy:
                best_energy = energy_an
                best_pts = pts_an
    
    # Final polish: more iterations of gradient descent with smaller lr
    if time.time() < deadline:
        pts_final, energy_final = _gradient_descent(best_pts, max_iter=5000, lr_init=0.001)
        if energy_final < best_energy:
            best_energy = energy_final
            best_pts = pts_final
    
    # Ensure all points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += np.random.randn(3) * 1e-6
                best_pts = _project_to_sphere(best_pts)
                break
    
    # Verify non-finite values
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
