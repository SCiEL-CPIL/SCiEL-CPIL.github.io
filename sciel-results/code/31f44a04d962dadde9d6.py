"""Rewritten program: multi-restart gradient descent with adaptive step for log-energy minimization on S^2."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # safety margin before 150s kill


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs."""
    # pts: (N, 3)
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # avoid self-pairs and very small distances
    np.fill_diagonal(dist, 1.0)
    # mask upper triangle
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _log_energy_and_grad(pts):
    """Return energy and gradient of log(1/d) energy."""
    # grad of log(1/|xi-xj|) w.r.t. xi is (xi - xj)/|xi-xj|^2 * (-1) ... 
    # Actually d/dxi log(1/|xi-xj|) = -d/dxi log|xi-xj| = -(xi-xj)/|xi-xj|^2
    # But energy = sum log(1/d) = -sum log(d), so grad_energy = -grad_sum_log_d
    # grad of log|xi-xj| w.r.t. xi = (xi-xj)/|xi-xj|^2
    # So grad_energy w.r.t. xi = -sum_j (xi-xj)/|xi-xj|^2 for j != i
    # Wait: E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dxi = -sum_{j!=i} d/dxi log|xi-xj| = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # But we need to project onto tangent space of sphere.
    
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(i + 1, N):
            d_vec = pts[i] - pts[j]
            d2 = np.dot(d_vec, d_vec)
            if d2 < 1e-30:
                continue
            d = math.sqrt(d2)
            # contribution to grad_i: -d_vec / d^2 ... wait
            # E includes -log(d_ij), d/dxi (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
            # So grad_i gets -(xi-xj)/d^2, grad_j gets +(xi-xj)/d^2
            coeff = 1.0 / d2
            grad[i] -= d_vec * coeff
            grad[j] += d_vec * coeff
    
    # Project gradient onto tangent space: g_tangent = g - (g . x) * x
    for i in range(N):
        g_dot_x = np.dot(grad[i], pts[i])
        grad[i] -= g_dot_x * pts[i]
    
    # Compute energy
    energy = _log_energy(pts)
    return energy, grad


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(start_pts, time_limit, rng_seed):
    """Run gradient descent with adaptive step from start_pts."""
    pts = start_pts.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    lr = 0.01
    min_lr = 1e-8
    max_lr = 0.5
    patience = 50
    no_improve = 0
    
    start_time = time.time()
    
    for step in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _log_energy_and_grad(pts)
        
        if energy < best_energy - 1e-12:
            best_energy = energy
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience and step > 100:
            # Try to escape: reduce lr and add small perturbation
            lr *= 0.5
            if lr < min_lr:
                break
            pts = _project_to_sphere(pts + rng_seed.standard_normal((N, 3)) * 0.01)
            no_improve = 0
            continue
        
        # Adaptive step: take step, if energy increases, halve lr
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            # Increase lr slightly if improvement is good
            if new_energy < energy - 0.01:
                lr = min(lr * 1.1, max_lr)
        else:
            lr *= 0.5
            if lr < min_lr:
                break
    
    return best_pts, best_energy


def construct():
    start_total = time.time()
    
    # Multiple restarts with different initializations
    best_pts = None
    best_energy = float('inf')
    
    # Restart 1: Fibonacci spiral
    fib = _fibonacci_points(N)
    e, p = _optimize_from_start(fib, 40.0, np.random.default_rng(42))
    if e < best_energy:
        best_energy = e
        best_pts = p
    
    elapsed = time.time() - start_total
    remaining = DEADLINE - elapsed
    
    if remaining < 10:
        return best_pts
    
    # Restart 2: Random initialization
    rng = np.random.default_rng(123)
    rand_pts = _random_points(N, 123)
    e, p = _optimize_from_start(rand_pts, min(30.0, remaining * 0.3), rng)
    if e < best_energy:
        best_energy = e
        best_pts = p
    
    elapsed = time.time() - start_total
    remaining = DEADLINE - elapsed
    if remaining < 10:
        return best_pts
    
    # Restart 3: Another random
    rand_pts2 = _random_points(N, 777)
    e, p = _optimize_from_start(rand_pts2, min(30.0, remaining * 0.3), np.random.default_rng(777))
    if e < best_energy:
        best_energy = e
        best_pts = p
    
    elapsed = time.time() - start_total
    remaining = DEADLINE - elapsed
    if remaining < 10:
        return best_pts
    
    # Restart 4: Perturbed Fibonacci
    rng4 = np.random.default_rng(999)
    fib_pert = _project_to_sphere(fib + rng4.standard_normal((N, 3)) * 0.1)
    e, p = _optimize_from_start(fib_pert, min(30.0, remaining * 0.4), rng4)
    if e < best_energy:
        best_energy = e
        best_pts = p
    
    # Final check
    if time.time() - start_total > DEADLINE - 5:
        return best_pts
    
    # One more restart if time allows
    remaining = DEADLINE - (time.time() - start_total)
    if remaining > 15:
        rand_pts3 = _random_points(N, 31415)
        e, p = _optimize_from_start(rand_pts3, min(20.0, remaining * 0.6), np.random.default_rng(31415))
        if e < best_energy:
            best_energy = e
            best_pts = p
    
    return best_pts
