"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij)."""
    # Compute pairwise distances
    d = cdist(pts, pts)
    # Only upper triangle, exclude diagonal
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    # Avoid log(0) - shouldn't happen with valid points
    dists = np.maximum(dists, 1e-15)
    return -np.sum(np.log(dists))  # log(1/d) = -log(d)


def _energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. point positions.
    
    E = sum_{i<j} -log(d_ij)
    d_ij = |x_i - x_j|
    dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2  (for the -log(d) term)
    
    Wait: E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j>i} (-1/|x_i-x_j|) * (x_i-x_j)/|x_i-x_j| + sum_{j<i} (-1/|x_j-x_i|) * (x_i-x_j)/|x_j-x_i|
            = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            grad[i] -= diff / (d * d)
    return grad


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _spherical_grad(pts, egrad):
    """Project gradient onto tangent space of sphere."""
    # Tangent projection: g_tan = g - (g . x) * x
    return egrad - np.sum(egrad * pts, axis=1, keepdims=True) * pts


def _fibsphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(pts0, max_iter=5000, tol=1e-10):
    """Run gradient descent on the sphere with adaptive step size."""
    pts = pts0.copy()
    best_pts = pts0.copy()
    best_e = _energy(best_pts)
    
    # Use a simple adaptive step size gradient descent
    step = 0.01
    patience = 0
    max_patience = 200
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        e = _energy(pts)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Compute gradient
        egrad = _energy_grad(pts)
        tgrad = _spherical_grad(pts, egrad)
        
        # Normalize gradient for stable step
        gnorm = np.linalg.norm(tgrad)
        if gnorm < tol:
            break
        
        # Adaptive step: reduce if energy increases
        new_pts = pts + step * tgrad
        new_pts = _project_to_sphere(new_pts)
        new_e = _energy(new_pts)
        
        if new_e < e:
            pts = new_pts
            step = min(step * 1.05, 0.1)
        else:
            step *= 0.5
            if step < 1e-12:
                break
    
    return best_pts, best_e


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    # Start from Fibonacci sphere
    pts = _fibsphere(N)
    
    # Multi-stage optimization: coarse to fine
    # Stage 1: Moderate steps, many iterations
    pts, e1 = _optimize_from_start(pts, max_iter=3000, tol=1e-8)
    
    # Stage 2: Fine-tuning with smaller steps
    if time.time() < _DEADLINE - 10:
        pts, e2 = _optimize_from_start(pts, max_iter=3000, tol=1e-12)
        if e2 < e1:
            pts = pts
            e1 = e2
    
    # Final check: ensure points are valid
    pts = _project_to_sphere(pts)
    
    # Verify no points are too close
    d = cdist(pts, pts)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(d[iu])
    if min_dist < 1e-9:
        # Fallback to Fibonacci if something went wrong
        pts = _fibsphere(N)
    
    return pts
