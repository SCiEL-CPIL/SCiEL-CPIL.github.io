"""Rewritten program using Riemannian mirror-descent / gradient flow on the sphere."""
import numpy as np
import math

N = 150

def construct():
    # Start with Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute logarithmic energy
    def energy(pts):
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to 1 to avoid log(0)
        np.fill_diagonal(dists, 1.0)
        # Compute sum of log(1/d) for i<j
        # Use upper triangle
        iu = np.triu_indices(N, k=1)
        vals = dists[iu]
        # Clip to avoid log of values too close to zero
        vals = np.maximum(vals, 1e-15)
        return math.fsum(math.log(1.0 / v) for v in vals)
    
    # Compute gradient of energy with respect to points on the sphere
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x_k) x_k
    def gradient(pts):
        grad = np.zeros_like(pts)
        for k in range(N):
            diff = pts[k] - pts  # shape (N, 3)
            dists = np.linalg.norm(diff, axis=1)
            np.fill_diagonal(dists, 1.0)
            # Avoid division by zero
            mask = dists > 1e-15
            inv_d2 = np.zeros(N)
            inv_d2[mask] = 1.0 / (dists[mask] ** 2)
            grad_k = np.sum(diff * inv_d2[:, np.newaxis], axis=0)
            # Project onto tangent space
            proj = np.dot(grad_k, pts[k])
            grad[k] = grad_k - proj * pts[k]
        return grad
    
    # Riemannian gradient descent with exponential map
    # On the sphere, the exponential map from x in direction v (tangent) is:
    # exp_x(v) = x cos(|v|) + (v/|v|) sin(|v|)
    def exponential_map(pts, v, step):
        """Apply Riemannian gradient step using exponential map."""
        new_pts = np.zeros_like(pts)
        for k in range(N):
            vk = step * v[k]
            norm_vk = np.linalg.norm(vk)
            if norm_vk < 1e-15:
                new_pts[k] = pts[k]
            else:
                # Normalize vk to get direction
                vk_dir = vk / norm_vk
                # Exponential map on unit sphere
                new_pts[k] = pts[k] * np.cos(norm_vk) + vk_dir * np.sin(norm_vk)
                # Re-normalize for numerical stability
                new_pts[k] /= np.linalg.norm(new_pts[k])
        return new_pts
    
    # Adaptive step size with backtracking
    best_pts = pts.copy()
    best_e = energy(pts)
    
    step = 0.01
    max_iter = 5000
    patience = 200
    no_improve = 0
    
    for it in range(max_iter):
        grad = gradient(pts)
        
        # Try the step
        new_pts = exponential_map(pts, grad, step)
        new_e = energy(new_pts)
        
        if new_e < best_e - 1e-12:
            pts = new_pts
            best_pts = new_pts.copy()
            best_e = new_e
            no_improve = 0
            # Occasionally increase step size if making good progress
            if it % 100 == 0 and step < 0.1:
                step *= 1.05
        else:
            # Backtrack: reduce step size
            step *= 0.7
            no_improve += 1
            if no_improve > patience:
                break
            if step < 1e-8:
                break
        
        # Periodically re-normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    
    # Final normalization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts
