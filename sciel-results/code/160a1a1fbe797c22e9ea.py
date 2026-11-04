"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import numpy as np
import time
import math

N = 150
DEADLINE = 145.0  # safety margin before 150s kill

def _energy_and_grad(pts):
    """Compute log-energy sum_{i<j} log(1/|x_i-x_j|) and gradient."""
    # pts: (N,3)
    # pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N,N)
    # avoid self and very small distances
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(diff, 0.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # For each pair (i,j), contribution to grad_i is (x_i - x_j)/d^2
    # and to grad_j is (x_j - x_i)/d^2 = -(x_i - x_j)/d^2
    # So grad = sum over j of (x_i - x_j)/d_ij^2
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    # grad_i = sum_j (x_i - x_j) * inv_d2[i,j]
    # = x_i * sum_j inv_d2[i,j] - sum_j x_j * inv_d2[i,j]
    row_sum = np.sum(inv_d2, axis=1)  # (N,)
    # sum_j x_j * inv_d2[i,j] = inv_d2 @ pts  (N,N)@(N,3) -> (N,3)
    weighted_sum = inv_d2 @ pts  # (N,3)
    grad = pts * row_sum[:, np.newaxis] - weighted_sum  # (N,3)
    
    # Project gradient onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1)  # (N,)
    grad_tangent = grad - dot[:, np.newaxis] * pts
    return energy, grad_tangent

def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize(pts, time_limit, rng, verbose=False):
    """Gradient descent with adaptive step size on the sphere."""
    pts = pts.copy()
    best_energy, best_pts = _energy_and_grad(pts)[0], pts.copy()
    lr = 1e-3
    min_lr = 1e-10
    patience = 0
    max_patience = 2000
    
    t0 = time.time()
    it = 0
    while time.time() - t0 < time_limit:
        energy, grad = _energy_and_grad(pts)
        it += 1
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Adaptive step: use gradient magnitude to scale
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        # Try step
        new_pts = pts - lr * grad
        # Renormalize to sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            # Increase lr slightly if improving
            lr = min(lr * 1.05, 1e-1)
        else:
            # Decrease lr
            lr = max(lr * 0.5, min_lr)
            if lr <= min_lr:
                break
        
        if verbose and it % 1000 == 0:
            print(f"  it={it}, E={energy:.10f}, lr={lr:.2e}")
    
    return best_pts, best_energy

def _polish(pts, time_limit):
    """High-precision Newton-like polish."""
    pts = pts.copy()
    t0 = time.time()
    energy, grad = _energy_and_grad(pts)
    
    # Use a very small step with line search
    lr = 1e-6
    for it in range(5000):
        if time.time() - t0 > time_limit:
            break
        energy, grad = _energy_and_grad(pts)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break
        
        # Backtracking line search
        alpha = 1.0
        for _ in range(20):
            new_pts = pts - lr * alpha * grad
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / norms
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                break
            alpha *= 0.5
        else:
            break
        
        pts = new_pts
        energy = new_energy
    
    return pts, energy

def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = np.inf
    
    # Strategy: multiple restarts with decreasing time budget
    # First: Fibonacci init with long optimization
    # Then: random inits with shorter budgets
    
    # Phase 1: Fibonacci spiral
    pts0 = _fibonacci_init()
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 5:
        pts1, e1 = _optimize(pts0, remaining * 0.4, rng)
        if e1 < best_energy:
            best_energy = e1
            best_pts = pts1
    
    # Phase 2: Several random restarts
    n_restarts = 5
    elapsed = time.time() - t_start
    remaining = DEADLINE - elapsed
    if remaining > 3:
        time_per_restart = remaining * 0.5 / n_restarts
        for r in range(n_restarts):
            if time.time() - t_start > DEADLINE - 3:
                break
            pts_r = _random_init(rng)
            pts_opt, e_r = _optimize(pts_r, time_per_restart, rng)
            if e_r < best_energy:
                best_energy = e_r
                best_pts = pts_opt
    
    # Phase 3: Polish the best
    if best_pts is not None:
        elapsed = time.time() - t_start
        remaining = DEADLINE - elapsed
        if remaining > 1:
            best_pts, best_energy = _polish(best_pts, remaining)
    
    # Final safety check
    if best_pts is None:
        best_pts = _fibonacci_init()
    
    # Ensure valid: normalize, check finite, check no points too close
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        # Perturb slightly
        rng2 = np.random.default_rng(0)
        best_pts = best_pts + rng2.standard_normal((N, 3)) * 1e-6
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
