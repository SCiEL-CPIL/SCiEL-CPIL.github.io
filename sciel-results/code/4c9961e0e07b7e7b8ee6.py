"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time


def _pairwise_distances(points):
    """Compute pairwise chordal distances for points on unit sphere."""
    # points: (N, 3)
    # Use the identity |x_i - x_j|^2 = 2 - 2 x_i . x_j for unit vectors
    dots = points @ points.T  # (N, N)
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 0.0, 4.0, out=d2)
    d = np.sqrt(d2)
    return d


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = _pairwise_distances(points)
    # Upper triangle, excluding diagonal
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    # Avoid log(0)
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    return fsum(vals.tolist())


def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points (in R^3).
    Energy = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    d2 = 2.0 - 2.0 * (points @ points.T)
    np.clip(d2, 1e-30, 4.0, out=d2)
    d = np.sqrt(d2)
    # For each pair (i,j), contribution to grad_i is (x_i - x_j)/d^2
    # grad_i = sum_j (x_i - x_j)/d_ij^2
    # We can compute this via matrix operations
    # M_ij = 1/d_ij^2 for i != j, 0 for i == j
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    # grad_i = M_i @ x_i - sum_j M_ij x_j
    # sum_j M_ij x_j = M @ points
    grad = M @ points
    # grad_i = M_ii * x_i - (M @ points)_i but M_ii=0
    # Actually: grad_i = sum_j M_ij (x_i - x_j) = x_i * sum_j M_ij - sum_j M_ij x_j
    row_sums = M.sum(axis=1, keepdims=True)
    grad = row_sums * points - grad
    return grad


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _project_gradient(points, grad):
    """Project gradient onto tangent space of unit sphere.
    Tangent space at x: {v : x . v = 0}
    Projection: v - (x . v) x
    """
    dots = (points * grad).sum(axis=1, keepdims=True)
    return grad - dots * points


def _fibonacci_spiral(N):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(N, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _gradient_descent(points, lr, n_iters, momentum=0.9, lr_decay=1.0, min_dist=1e-9):
    """Gradient descent on the sphere with momentum and adaptive learning rate."""
    points = points.copy()
    points = _normalize(points)
    v = np.zeros_like(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    for it in range(n_iters):
        # Compute energy and gradient
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        grad = _project_gradient(points, grad)
        
        # Adaptive learning rate based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Momentum update
        v = momentum * v - lr * lr_decay * grad / (grad_norm + 1e-15)
        points = points + v
        
        # Normalize to sphere
        points = _normalize(points)
        
        # Check for close points
        d = _pairwise_distances(points)
        if d.min() < min_dist:
            # Push apart close points
            iu = np.triu_indices(N, k=1)
            close = d[iu] < min_dist * 10
            for idx in range(len(close)):
                if close[idx]:
                    i, j = iu[0][idx], iu[1][idx]
                    # Separate them
                    mid = (points[i] + points[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-15:
                        mid = mid / mid_norm
                    perp = points[i] - points[j]
                    perp = perp - np.dot(perp, mid) * mid
                    perp_norm = np.linalg.norm(perp)
                    if perp_norm < 1e-15:
                        perp = np.array([1.0, 0.0, 0.0])
                        perp = perp - np.dot(perp, mid) * mid
                        perp_norm = np.linalg.norm(perp)
                    perp = perp / perp_norm
                    points[i] = _normalize(points[i] + 0.01 * perp)
                    points[j] = _normalize(points[j] - 0.01 * perp)
        
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Decay learning rate
        lr_decay = max(0.999, 1.0 - it / (2.0 * n_iters))
    
    return best_points, best_energy


def _anneal(points, n_restarts=5, lr_start=0.01, lr_end=0.0001, n_iters=2000, seed=0):
    """Multi-restart gradient descent with simulated annealing-like schedule."""
    rng = np.random.default_rng(seed)
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    for restart in range(n_restarts):
        if restart == 0:
            pts = points.copy()
        else:
            # Perturb from best
            noise = rng.standard_normal((N, 3)) * (0.05 * (1.0 + restart * 0.5))
            pts = _normalize(points + noise)
        
        # Schedule
        lr = lr_start
        for phase in range(3):
            n_phase = n_iters // 3
            lr_phase = lr_start * (lr_end / lr_start) ** (phase / 3.0)
            pts, energy = _gradient_descent(pts, lr_phase, n_phase, momentum=0.9)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
        
        # Small final polish
        pts, energy = _gradient_descent(pts, lr_end, n_iters // 5, momentum=0.99)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    return best_points, best_energy


def construct():
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    points = _normalize(points)
    
    # Multi-scale optimization: coarse to fine
    # Phase 1: large steps, few iterations
    points, e1 = _gradient_descent(points, lr=0.005, n_iters=500, momentum=0.9)
    
    # Phase 2: medium steps
    points, e2 = _gradient_descent(points, lr=0.001, n_iters=1000, momentum=0.9)
    
    # Phase 3: fine steps
    points, e3 = _gradient_descent(points, lr=0.0001, n_iters=2000, momentum=0.95)
    
    # Multi-restart to escape local minima
    points, e4 = _anneal(points, n_restarts=3, lr_start=0.002, lr_end=0.00005, n_iters=1500, seed=42)
    
    # Final polish
    points, e5 = _gradient_descent(points, lr=0.00005, n_iters=1000, momentum=0.99)
    
    # Ensure valid output
    points = _normalize(points)
    
    # Verify no points are too close
    d = _pairwise_distances(points)
    if d.min() < 1e-9:
        # Fallback to Fibonacci spiral
        points = _fibonacci_spiral(N)
        points = _normalize(points)
    
    return points
# EVOLVE-BLOCK-END
