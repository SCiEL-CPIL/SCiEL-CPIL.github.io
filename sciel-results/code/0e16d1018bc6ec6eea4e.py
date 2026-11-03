"""Minimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|xi-xj|) and gradient on sphere.
    Returns (energy, grad) where grad is the ambient gradient projected to tangent space."""
    n = pts.shape[0]
    # Pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: d/dxi [ -log|xi-xj| ] = -(xi-xj)/|xi-xj|^2  for each j != i
    # So grad_i = sum_{j!=i} (xi-xj) / |xi-xj|^2  ... wait, let's re-derive.
    # E = sum_{i<j} -log|xi-xj|
    # dE/dxi = sum_{j!=i} - (1/|xi-xj|) * (xi-xj)/|xi-xj| = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # So ambient grad_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # But we want to minimize E, so the direction of steepest descent is -grad.
    # For Riemannian optimization, we project the negative gradient onto the tangent space.
    
    # Compute sum_{j!=i} (xi-xj)/|xi-xj|^2 for each i
    # diff[i,j] = xi - xj, dist[i,j] = |xi-xj|
    # term[i,j] = diff[i,j] / dist[i,j]^2
    # But dist[i,i]=1 (set above), and we want to exclude j=i.
    # Actually, let's compute the gradient properly.
    
    # grad_ambient[i] = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # = sum_{j!=i} (xj-xi)/|xi-xj|^2
    
    # Let's compute S[i] = sum_{j!=i} (xj-xi)/|xi-xj|^2
    # S[i] = sum_{j!=i} xj/|xi-xj|^2 - xi * sum_{j!=i} 1/|xi-xj|^2
    
    # More efficient: 
    # inv_dist2 = 1/dist^2, with diagonal set to 0
    inv_dist2 = np.zeros_like(dist)
    np.fill_diagonal(dist, 1.0)  # reset
    mask = ~np.eye(n, dtype=bool)
    inv_dist2[mask] = 1.0 / (dist[mask] ** 2)
    
    # S[i] = sum_j xj * inv_dist2[i,j] - xi * sum_j inv_dist2[i,j]
    # = pts.T @ inv_dist2 - pts * (inv_dist2.sum(axis=1, keepdims=True))
    S = pts.T @ inv_dist2 - pts * (inv_dist2.sum(axis=1, keepdims=True))
    
    # Ambient gradient of E is -S (since dE/dxi = -S[i])
    # We want to minimize, so Riemannian gradient = projection of ambient grad onto tangent space
    # Riemannian grad_i = -S[i] - (<-S[i], xi> * xi) = -S[i] + (<S[i], xi> * xi)
    # Wait: projection of vector v onto tangent space at xi is v - <v,xi>xi
    # So Riemannian grad = proj(-S) = -S - <-S, xi> xi = -S + <S, xi> xi
    
    dot_S_xi = np.sum(S * pts, axis=1, keepdims=True)
    rgrad = -S + dot_S_xi * pts
    
    return energy, rgrad


def _riemannian_cg(pts, max_iter=500, tol=1e-12):
    """Riemannian conjugate gradient on the sphere for minimizing log energy."""
    n = pts.shape[0]
    energy, rgrad = _energy_and_grad(pts)
    
    if np.linalg.norm(rgrad) < tol:
        return pts, energy
    
    d = -rgrad.copy()
    # Project d onto tangent space (it should already be, but ensure)
    d = d - np.sum(d * pts, axis=1, keepdims=True) * pts
    
    beta = 0.0
    for k in range(max_iter):
        # Line search: minimize f(exp_{pts}(t*d)) for t in [0, 1]
        # Use a simple backtracking line search
        alpha = 0.1
        for _ in range(50):
            # Geodesic step: exp_x(t*d) = x*cos(t*|d|) + d/|d|*sin(t*|d|)
            d_norm = np.linalg.norm(d, axis=1, keepdims=True)
            d_norm = np.maximum(d_norm, 1e-15)
            t_d = alpha * d_norm
            new_pts = pts * np.cos(t_d) + (d / d_norm) * np.sin(t_d)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                break
            alpha *= 0.5
        else:
            # If line search fails, break
            break
        
        pts = new_pts
        energy = new_energy
        rgrad_new, _ = _energy_and_grad(pts)
        rgrad_new = rgrad_new - np.sum(rgrad_new * pts, axis=1, keepdims=True) * pts
        
        if np.linalg.norm(rgrad_new) < tol:
            rgrad = rgrad_new
            break
        
        # Polak-Ribiere beta
        beta = np.sum(rgrad_new * (rgrad_new - rgrad)) / max(np.sum(rgrad * rgrad), 1e-30)
        beta = max(0.0, beta)
        d = -rgrad_new + beta * d
        # Project onto tangent space
        d = d - np.sum(d * pts, axis=1, keepdims=True) * pts
        rgrad = rgrad_new
    
    return pts, energy


def _basin_hopping_step(pts, energy, temperature=0.15):
    """Perturb some points and accept if energy decreases or with Metropolis criterion."""
    n = pts.shape[0]
    # Choose number of points to perturb
    n_perturb = max(1, int(0.05 * n))
    indices = np.random.choice(n, n_perturb, replace=False)
    
    # Create perturbed points
    new_pts = pts.copy()
    for idx in indices:
        # Random small perturbation on the sphere
        delta = np.random.randn(3) * temperature
        # Project to tangent space at pts[idx]
        dot = np.dot(delta, pts[idx])
        delta = delta - dot * pts[idx]
        # Exponential map
        d_norm = np.linalg.norm(delta)
        if d_norm > 1e-15:
            new_pts[idx] = pts[idx] * np.cos(d_norm) + (delta / d_norm) * np.sin(d_norm)
        # Ensure unit norm
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
    
    new_energy, _ = _energy_and_grad(new_pts)
    
    # Metropolis acceptance
    if new_energy < energy or np.random.rand() < np.exp(-(new_energy - energy)):
        return new_pts, new_energy
    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Phase 1: Riemannian CG polish from initial guess
    if time.time() < deadline - 5.0:
        pts, energy = _riemannian_cg(pts, max_iter=300, tol=1e-12)
        if energy < best_energy:
            best_pts = pts.copy()
            best_energy = energy
    
    # Phase 2: Basin hopping
    np.random.seed(42)
    temperature = 0.15
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    while time.time() < deadline - 10.0:
        # Basin hopping step
        current_pts, current_energy = _basin_hopping_step(current_pts, current_energy, temperature)
        
        # Polish with Riemannian CG
        polished_pts, polished_energy = _riemannian_cg(current_pts, max_iter=200, tol=1e-12)
        
        if polished_energy < best_energy:
            best_pts = polished_pts.copy()
            best_energy = polished_energy
            current_pts = polished_pts.copy()
            current_energy = polished_energy
        
        # Occasionally cool down
        if np.random.rand() < 0.1:
            temperature = max(0.05, temperature * 0.9)
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2.0:
        final_pts, final_energy = _riemannian_cg(best_pts, max_iter=500, tol=1e-14)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
    
    # Ensure unit norm and valid output
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts

# EVOLVE-BLOCK-END
