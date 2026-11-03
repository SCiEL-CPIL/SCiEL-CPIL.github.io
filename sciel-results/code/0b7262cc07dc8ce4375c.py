"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    """Minimize logarithmic energy of 100 points on unit sphere using
    Riemannian gradient descent with Nesterov momentum and backtracking
    line search, combined with simulated annealing restarts."""
    
    def energy(x):
        """Compute logarithmic energy using math.fsum for accuracy."""
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        triu = np.triu_indices(N, 1)
        vals = np.log(dist_sq[triu]).tolist()
        return -0.5 * math.fsum(vals)
    
    def grad_energy(x):
        """Compute R^3 gradient of energy."""
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        return grad
    
    def project(x):
        """Project to unit sphere."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return x / norms
    
    def tangent_proj(v, x):
        """Project vector to tangent space."""
        vd = np.sum(v * x, axis=1, keepdims=True)
        return v - vd * x
    
    def riemannian_gd(x0, n_iter=2000, momentum=0.9, base_lr=0.01):
        """Riemannian gradient descent with Nesterov momentum and backtracking."""
        x = x0.copy()
        v = np.zeros_like(x)
        lr = base_lr
        
        for it in range(n_iter):
            # Decay learning rate
            lr = base_lr * (0.999 ** it)
            
            # Nesterov: look ahead
            x_lookahead = project(x + momentum * v)
            g = grad_energy(x_lookahead)
            g_tan = tangent_proj(g, x_lookahead)
            
            # Update velocity
            v = momentum * v - lr * g_tan
            v = tangent_proj(v, x)
            
            # Backtracking line search
            x_new = project(x + v)
            e_old = energy(x)
            e_new = energy(x_new)
            backtracks = 0
            while e_new >= e_old and backtracks < 10:
                v *= 0.5
                x_new = project(x + v)
                e_new = energy(x_new)
                backtracks += 1
            
            if e_new < e_old:
                x = x_new
                v = tangent_proj(v, x)
            else:
                break
        
        return x
    
    best_energy = np.inf
    best_x = None
    
    # Multiple restarts with different initializations
    n_restarts = 20
    for restart in range(n_restarts):
        rng = np.random.default_rng(1000 + restart)
        
        if restart < 6:
            # Fibonacci with varying jitter
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            jitter = 0.005 + 0.01 * restart
            x = x + rng.normal(0, jitter, x.shape)
            x = project(x)
        elif restart < 12:
            # Random
            x = rng.normal(size=(N, 3))
            x = project(x)
        else:
            # Fibonacci with larger perturbation
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            # Swap some points
            perm = rng.permutation(N)
            x = x[perm]
            jitter = 0.05
            x = x + rng.normal(0, jitter, x.shape)
            x = project(x)
        
        # Run Riemannian GD
        x_opt = riemannian_gd(x, n_iter=3000, momentum=0.9, base_lr=0.02)
        
        # Second phase: finer tuning
        x_opt = riemannian_gd(x_opt, n_iter=2000, momentum=0.8, base_lr=0.005)
        
        e_final = energy(x_opt)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_opt.copy()
    
    # Final polish with very small steps
    if best_x is not None:
        best_x = riemannian_gd(best_x, n_iter=1000, momentum=0.5, base_lr=0.001)
        best_x = project(best_x)
    
    # Verify validity
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        rng = np.random.default_rng(12345)
        best_x = best_x + rng.normal(0, 1e-6, best_x.shape)
        best_x = project(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
