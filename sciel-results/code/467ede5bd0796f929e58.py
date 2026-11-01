"""Optimized logarithmic energy minimization for 150 points on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE_S = 115.0  # safety margin before 120s limit


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i-x_j|) and gradient.
    
    Energy = -sum_{i<j} log(|x_i - x_j|)
    Gradient wrt x_k: sum_{j!=k} (x_k - x_j) / |x_k - x_j|^2
    """
    n = pts.shape[0]
    # Pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point k, grad_k = sum_{j!=k} (x_k - x_j) / |x_k - x_j|^2
    # = sum_{j!=k} diff[k,j] / dist[k,j]^2
    # diff[k,j] / dist[k,j]^2 = diff[k,j] * (1/dist[k,j]^2)
    inv_dist2 = np.zeros_like(dist)
    np.fill_diagonal(inv_dist2, 0.0)
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[k] = sum_j diff[k,j] * inv_dist2[k,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _halton_sphere():
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = np.zeros((N, 3))
    for i in range(N):
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        # Use inverse CDF for uniform on sphere
        theta = 2.0 * np.pi * u
        cos_phi = 2.0 * v - 1.0
        sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
        pts[i] = [sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi]
    return pts


def _sphere_lbfgs_wrapper(pts0, time_budget):
    """
    Optimizes points on the sphere using L-BFGS-B with projection.
    This approximates Riemannian optimization by projecting the state and gradient.
    """
    def fun_and_grad(x):
        # Reshape and project
        p = x.reshape(N, 3)
        p = _project(p)
        
        # Compute energy and gradient
        e, g = _energy_and_grad(p)
        
        # Project gradient to tangent space: g_tan = g - (g . p) * p
        # Since p is unit vector, g . p is the normal component
        proj = np.sum(g * p, axis=1, keepdims=True)
        g_tan = g - proj * p
        
        return e, g_tan.flatten()

    x0 = pts0.flatten()
    
    # Use L-BFGS-B
    try:
        res = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxls': 50,
                'maxfun': 100000
            }
        )
        pts_res = _project(res.x.reshape(N, 3))
        e_res, _ = _energy_and_grad(pts_res)
        return pts_res, e_res
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]


def construct():
    t0 = time.time()
    
    # Generate Halton start
    start_pts = _halton_sphere()
    
    best_pts = start_pts
    best_energy = _energy_and_grad(start_pts)[0]
    
    # Run optimization with time budget
    elapsed = time.time() - t0
    remaining = DEADLINE_S - elapsed
    
    if remaining > 10:
        # Primary optimization
        time_budget = remaining * 0.8
        pts, energy = _sphere_lbfgs_wrapper(start_pts, time_budget)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
        
        # Check time again
        elapsed = time.time() - t0
        remaining = DEADLINE_S - elapsed
        
        if remaining > 15:
            # Try a second start: rotate the Halton sphere
            angle = np.pi / 2.0
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            start_pts2 = start_pts @ R.T
            start_pts2 = _project(start_pts2)
            
            time_budget2 = remaining * 0.8
            pts2, energy2 = _sphere_lbfgs_wrapper(start_pts2, time_budget2)
            
            if energy2 < best_energy:
                best_energy = energy2
                best_pts = pts2
        
        # Final polish if time remains
        elapsed = time.time() - t0
        remaining = DEADLINE_S - elapsed
        if remaining > 5:
            time_budget3 = remaining * 0.9
            pts3, energy3 = _sphere_lbfgs_wrapper(best_pts, time_budget3)
            if energy3 < best_energy:
                best_energy = energy3
                best_pts = pts3

    # Ensure validity
    best_pts = _project(best_pts)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge point j slightly
                noise = np.random.default_rng(j).normal(size=3) * 1e-6
                best_pts[j] = best_pts[j] + noise
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
