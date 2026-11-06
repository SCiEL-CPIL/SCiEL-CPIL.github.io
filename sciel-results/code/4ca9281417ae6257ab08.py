"""Optimize 100 points on the unit sphere to minimize logarithmic energy."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/d_ij)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e10
            total += -math.log(d)
    return total

def _energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. point coordinates."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                continue
            # d/dx_i log(1/d) = -1/d * d_vec/d = -d_vec/d^2
            # Actually: E = sum log(1/d) = -sum log(d)
            # dE/d(pts[i]) = -d(log d)/d(pts[i]) = -(pts[i]-pts[j])/d^2
            # dE/d(pts[j]) = +(pts[i]-pts[j])/d^2
            inv_d2 = 1.0 / (d * d)
            grad[i] += -d_vec * inv_d2
            grad[j] += d_vec * inv_d2
    return grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_from_start(start_pts, time_limit):
    """Run gradient descent with line search on the sphere from start_pts."""
    pts = _project_to_sphere(start_pts.copy())
    best_pts = pts.copy()
    best_e = _energy(pts)
    
    t0 = time.time()
    lr = 0.01
    step = 0
    
    while time.time() - t0 < time_limit:
        # Compute gradient
        g = _energy_grad(pts)
        
        # Project gradient onto tangent space
        g_tangent = g - np.einsum('ni,ni->n', g, pts) * pts
        
        # Check convergence
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < 1e-12:
            break
        
        # Adaptive step size
        step += 1
        if step % 50 == 0:
            lr *= 0.9
        if step % 100 == 0:
            lr = max(lr * 1.2, 0.001)
        
        # Gradient step
        new_pts = pts - lr * g_tangent
        new_pts = _project_to_sphere(new_pts)
        
        # Check energy
        new_e = _energy(new_pts)
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Backtrack
            lr *= 0.5
            new_pts = pts - lr * g_tangent
            new_pts = _project_to_sphere(new_pts)
            new_e = _energy(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts.copy()
                pts = new_pts
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
    
    return best_pts, best_e

def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral()
    best_pts = pts.copy()
    best_e = _energy(pts)
    
    # Phase 1: Coarse gradient descent
    remaining = DEADLINE - (time.time() - t_start) - 20.0  # reserve 20s for final polish
    if remaining > 5:
        pts, e = _optimize_from_start(pts, remaining)
        if e < best_e:
            best_e = e
            best_pts = pts
    
    # Phase 2: Local refinement with smaller steps
    remaining = DEADLINE - (time.time() - t_start) - 10.0
    if remaining > 5:
        # Try multiple small perturbations
        for attempt in range(3):
            if time.time() - t_start > DEADLINE - 10:
                break
            # Add small noise
            noise = np.random.randn(N, 3) * 0.01
            perturbed = _project_to_sphere(best_pts + noise)
            pts, e = _optimize_from_start(perturbed, min(remaining / (3 - attempt), 10.0))
            if e < best_e:
                best_e = e
                best_pts = pts
    
    # Final: ensure valid output
    best_pts = _project_to_sphere(best_pts)
    return best_pts
