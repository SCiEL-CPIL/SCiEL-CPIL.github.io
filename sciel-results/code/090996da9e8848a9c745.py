"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time

N = 100

def _pairwise_distances(points):
    """Compute pairwise chordal distances for points on unit sphere."""
    dots = points @ points.T
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 0.0, 4.0, out=d2)
    d = np.sqrt(d2)
    return d

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = _pairwise_distances(points)
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    return fsum(vals.tolist())

def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points (in R^3)."""
    d2 = 2.0 - 2.0 * (points @ points.T)
    np.clip(d2, 1e-30, 4.0, out=d2)
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    grad = row_sums * points - (M @ points)
    return grad

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _project_gradient(points, grad):
    dots = (points * grad).sum(axis=1, keepdims=True)
    return grad - dots * points

def _fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(N, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _gradient_descent(points, n_iters=2000, lr_init=0.01, lr_end=0.0001):
    """Gradient descent with backtracking line search."""
    points = _normalize(points.copy())
    energy = _log_energy(points)
    best_points = points.copy()
    best_energy = energy
    
    for it in range(n_iters):
        grad = _log_energy_grad(points)
        grad = _project_gradient(points, grad)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        # Adaptive learning rate
        lr = lr_init * (lr_end / lr_init) ** (it / n_iters)
        
        # Backtracking line search
        step = lr
        current_energy = energy
        improved = False
        for _ in range(20):
            new_points = _normalize(points - step * grad)
            new_energy = _log_energy(new_points)
            if new_energy < current_energy:
                points = new_points
                energy = new_energy
                improved = True
                break
            step *= 0.5
            
        if not improved:
            # If no improvement, try a small random kick to escape local minima
            if it % 100 == 0:
                rng = np.random.default_rng(it)
                noise = rng.standard_normal((N, 3)) * 0.01
                points = _normalize(points + noise)
                energy = _log_energy(points)
            else:
                break
                
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
    return best_points, best_energy

def construct():
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    points = _normalize(points)
    
    # Optimization
    points, e1 = _gradient_descent(points, n_iters=3000, lr_init=0.05, lr_end=0.0001)
    
    # Try a few random restarts
    for seed in [1, 2, 3]:
        rand_pts = _random_points(N, seed=seed)
        rand_pts, e_rand = _gradient_descent(rand_pts, n_iters=1000, lr_init=0.05, lr_end=0.0001)
        if e_rand < e1:
            points = rand_pts
            e1 = e_rand
            
    # Final polish
    points, e_final = _gradient_descent(points, n_iters=2000, lr_init=0.001, lr_end=0.00001)
    
    points = _normalize(points)
    d = _pairwise_distances(points)
    if d.min() < 1e-9:
        points = _fibonacci_spiral(N)
        points = _normalize(points)
        
    return points
# EVOLVE-BLOCK-END
