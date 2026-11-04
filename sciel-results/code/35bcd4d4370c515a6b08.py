"""Rewritten: optimize log-energy on the sphere via projected gradient descent with restarts."""
import time
import numpy as np
from math import fsum

N = 150
HARD_LIMIT = 150.0
SAFETY = 10.0  # seconds before the hard limit


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/||x_i - x_j||)."""
    N = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    triu_idx = np.triu_indices(N, k=1)
    d_vals = np.sqrt(d2[triu_idx])
    d_vals = np.maximum(d_vals, 1e-15)
    return -np.sum(np.log(d_vals))


def _energy_grad(pts):
    """Compute gradient of logarithmic energy. grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2."""
    N = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(d2, 1.0)
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    contrib = diff * inv_d2[:, :, np.newaxis]  # (N, N, 3)
    grad = np.sum(contrib, axis=1)  # (N, 3)
    return grad


def _project_tangent(pts, grad):
    """Project gradient onto tangent plane at each point: g - (g·x_i) x_i."""
    dots = np.einsum('ij,ij->i', grad, pts)
    return grad - dots[:, None] * pts


def _fibonacci_sphere():
    """Generate well-distributed points on sphere using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate random points on sphere."""
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _grad_descent(pts, rng, budget, lr_init=1e-2):
    """Projected gradient descent with adaptive step size.
    
    Args:
        pts: initial points (N, 3)
        rng: random number generator
        budget: absolute time deadline (time.time() value)
        lr_init: initial learning rate
    Returns:
        (best_points, best_energy)
    """
    pts = pts.copy()
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    best = pts.copy()
    best_e = _energy(best)
    lr = lr_init
    prev_grad = None
    prev_pts = None
    n = 0
    
    while time.time() < budget:
        g = _energy_grad(pts)
        gt = _project_tangent(pts, g)
        gnorm = np.linalg.norm(gt)
        if gnorm < 1e-12:
            break
        
        # Barzilai-Borwein step size if we have previous gradient
        if prev_grad is not None:
            y = gt - prev_grad  # gradient difference
            s = pts - prev_pts  # point difference
            ys = np.dot(y.flatten(), s.flatten())
            sy = np.dot(s.flatten(), y.flatten())
            if abs(ys) > 1e-15:
                # BB step: alpha = <s, s> / <s, y>
                ss = np.dot(s.flatten(), s.flatten())
                bb_lr = ss / ys
                if bb_lr > 0:
                    lr = np.clip(bb_lr, 1e-8, 0.5)
        
        step = gt * lr
        pts_new = pts + step
        pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
        e_new = _energy(pts_new)
        
        if e_new < best_e:
            best = pts_new.copy()
            best_e = e_new
            # Slightly increase lr on success
            lr = min(lr * 1.05, 0.5)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
        
        prev_pts = pts.copy()
        prev_grad = gt.copy()
        pts = pts_new
        n += 1
        
        if n % 50 == 0 and time.time() > budget:
            break
    
    return best, best_e


def construct():
    t0 = time.time()
    deadline = t0 + HARD_LIMIT - SAFETY  # absolute time deadline
    rng = np.random.default_rng(12345)
    best_pts = None
    best_e = np.inf

    # Multiple initializations: Fibonacci + random restarts
    inits = [_fibonacci_sphere()]
    for _ in range(3):
        inits.append(_random_sphere(rng))

    # Phase 1: Main optimization with each initialization
    n_inits = len(inits)
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        # Allocate time: more time for earlier restarts
        per_restart_time = remaining / (n_inits - i)
        per_restart_time = max(per_restart_time, 3.0)
        
        pts, e = _grad_descent(init, rng, time.time() + per_restart_time, lr_init=1e-2)
        if e < best_e:
            best_e = e
            best_pts = pts

    if best_pts is None:
        best_pts = _fibonacci_sphere()
        best_e = _energy(best_pts)

    # Phase 2: Polish with smaller learning rate from best configuration
    if time.time() < deadline - 2.0:
        polish_time = deadline - time.time() - 1.0
        if polish_time > 2.0:
            pts, e = _grad_descent(best_pts, rng, time.time() + polish_time, lr_init=1e-4)
            if e < best_e:
                best_e = e
                best_pts = pts

    # Phase 3: Final fine polish with very small steps
    if time.time() < deadline - 1.0:
        final_time = deadline - time.time() - 0.5
        if final_time > 1.0:
            pts, e = _grad_descent(best_pts, rng, time.time() + final_time, lr_init=1e-6)
            if e < best_e:
                best_e = e
                best_pts = pts

    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
