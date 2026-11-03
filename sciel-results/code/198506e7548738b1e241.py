"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def construct():
    t0 = time.time()
    deadline = 120.0  # safety margin before 150s limit
    
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    def energy_and_grad(pts):
        """Compute logarithmic energy and its gradient."""
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0) and division by zero on diagonal
        mask = ~np.eye(N, dtype=bool)
        dist_safe = np.where(mask, dist, 1.0)
        
        # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
        # We compute full sum then halve
        log_dist = np.log(dist_safe)
        energy = -0.5 * np.sum(log_dist[mask])
        
        # Gradient: d/dxi [-log|xi-xj|] = (xi-xj)/|xi-xj|^2 for j != i
        # grad_i = sum_{j != i} (xi - xj) / |xi - xj|^2
        # = sum_{j != i} diff[i,j] / dist[i,j]^2
        
        # diff[i,j] = xi - xj
        # dist[i,j]^2 = sum of squared components
        dist_sq = dist_safe ** 2
        dist_sq_safe = np.where(mask, dist_sq, 1.0)
        
        # grad[i] = sum_j (xi - xj) / |xi-xj|^2 for j != i
        # = sum_j diff[i,j] / dist_sq[i,j]
        grad = np.sum(diff / dist_sq_safe[:, :, np.newaxis], axis=1)
        
        return energy, grad
    
    # Get initial energy
    best_energy, _ = energy_and_grad(pts)
    best_pts = pts.copy()
    
    # Gradient descent with momentum
    momentum = np.zeros_like(pts)
    lr = 0.01
    beta = 0.9
    min_lr = 1e-6
    
    step_count = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - t0 < deadline:
        energy, grad = energy_and_grad(pts)
        
        # Update momentum
        momentum = beta * momentum - lr * grad
        pts_new = pts + momentum
        
        # Project back to unit sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check validity
        if not np.all(np.isfinite(pts_new)):
            lr *= 0.5
            continue
        
        # Check minimum distance
        diff = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        if dist.min() < 1e-9:
            lr *= 0.5
            continue
        
        energy_new, _ = energy_and_grad(pts_new)
        
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            no_improve_count = 0
            # Slightly increase learning rate if we're making good progress
            if step_count % 100 == 0 and lr < 0.05:
                lr *= 1.05
        else:
            no_improve_count += 1
            # Decrease learning rate when stuck
            if no_improve_count > 100:
                lr *= 0.9
                if lr < min_lr:
                    lr = min_lr
            
        pts = pts_new
        step_count += 1
        
        # If learning rate is very small and we've been stuck, try a restart
        if no_improve_count > max_no_improve and lr < 1e-5:
            # Try a few random restarts with perturbation
            for restart in range(5):
                pts_pert = best_pts + 0.01 * np.random.randn(N, 3)
                norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
                pts_pert = pts_pert / norms
                
                energy_pert, grad_pert = energy_and_grad(pts_pert)
                if energy_pert < best_energy:
                    best_energy = energy_pert
                    best_pts = pts_pert.copy()
                    pts = pts_pert.copy()
                    momentum = np.zeros_like(pts)
                    lr = 0.01
                    no_improve_count = 0
                    break
            
            if no_improve_count > max_no_improve * 2:
                break
    
    # Final polish: run more iterations with smaller learning rate
    if time.time() - t0 < deadline - 5.0:
        lr_final = 1e-6
        beta_final = 0.95
        momentum = np.zeros_like(pts)
        
        for _ in range(200):
            if time.time() - t0 > deadline:
                break
            energy, grad = energy_and_grad(pts)
            momentum = beta_final * momentum - lr_final * grad
            pts_new = pts + momentum
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            
            if np.all(np.isfinite(pts_new)):
                energy_new, _ = energy_and_grad(pts_new)
                if energy_new < best_energy:
                    best_energy = energy_new
                    best_pts = pts_new.copy()
            pts = pts_new
    
    # Ensure final result is valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
