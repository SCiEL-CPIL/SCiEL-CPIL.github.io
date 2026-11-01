import numpy as np
from math import fsum, log
from scipy.optimize import minimize

N = 150

def compute_energy_fsum(pts):
    """Energy with fsum for precision."""
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    d = np.sqrt(np.sum(diff**2, axis=1))
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0/d)
    return fsum(terms.tolist())

def energy_wrapper(x):
    """Wrapper for scipy minimize."""
    pts = x.reshape(N, 3)
    # Project to sphere to handle any drift during optimization
    norms = np.linalg.norm(pts, axis=1)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms[:, np.newaxis]
    return compute_energy_fsum(pts)

def construct():
    np.random.seed(42)
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initialize Distance Matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, 1.0)
    
    # Compute initial energy
    iu = np.triu_indices(N, k=1)
    current_energy = fsum(np.log(1.0 / D[iu]).tolist())
    best_energy = current_energy
    best_pts = pts.copy()
    
    # Simulated Annealing
    # Increased iterations due to faster move
    T0 = 0.1
    Tf = 0.001
    n_iter = 500000
    alpha = (Tf/T0)**(1.0/n_iter)
    T = T0
    
    for it in range(n_iter):
        T = max(T * alpha, Tf)
        
        k = np.random.randint(N)
        
        # Fast rotation using Rodrigues' formula
        theta = np.random.uniform(0.05, 0.5)
        axis = np.random.randn(3)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= norm_axis
        
        c = np.cos(theta)
        s = np.sin(theta)
        old_pt = pts[k]
        
        # Rodrigues: v' = v cos + (k x v) sin + k (k . v) (1 - cos)
        cross = np.cross(axis, old_pt)
        dot = np.dot(axis, old_pt)
        new_pt = old_pt * c + cross * s + axis * dot * (1.0 - c)
        
        norm_new = np.linalg.norm(new_pt)
        if norm_new < 1e-10:
            continue
        new_pt /= norm_new
        
        # Compute delta energy
        others_idx = np.arange(N)
        others_idx = others_idx[others_idx != k]
        
        d_old = D[k, others_idx]
        d_new = np.linalg.norm(new_pt - pts[others_idx], axis=1)
        
        d_old = np.maximum(d_old, 1e-15)
        d_new = np.maximum(d_new, 1e-15)
        
        # Use log(1/d) = -log(d)
        delta_E = np.sum(np.log(d_old)) - np.sum(np.log(d_new))
        
        # Metropolis
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
            pts[k] = new_pt
            current_energy += delta_E
            
            D[k, others_idx] = d_new
            D[others_idx, k] = d_new
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        # Periodic recompute to avoid drift
        if it % 10000 == 0:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(D, 1.0)
            current_energy = fsum(np.log(1.0 / D[iu]).tolist())
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

    # Polish with scipy.optimize.minimize
    # Use L-BFGS-B with a penalty term to enforce sphere constraint
    # This is often faster and more robust than manual gradient descent
    
    x0 = best_pts.flatten()
    
    def penalized_energy(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        penalty = 100.0 * np.sum((norms - 1.0)**2)
        # Project for energy calculation to ensure stability
        norms_safe = np.maximum(norms, 1e-15)
        pts_proj = pts / norms_safe[:, np.newaxis]
        return compute_energy_fsum(pts_proj) + penalty
    
    # Gradient of penalized energy
    def penalized_grad(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        norms_safe = np.maximum(norms, 1e-15)
        pts_proj = pts / norms_safe[:, np.newaxis]
        
        # Gradient of energy w.r.t pts_proj
        grads_proj = np.zeros_like(pts_proj)
        for k in range(N):
            others = np.delete(pts_proj, k, axis=0)
            diff = pts_proj[k][np.newaxis, :] - others
            dists_sq = np.sum(diff**2, axis=1)
            dists_sq = np.maximum(dists_sq, 1e-30)
            # d/dx log(1/||x-y||) = - (x-y) / ||x-y||^2
            grads_proj[k] = -np.sum(diff / dists_sq[:, np.newaxis], axis=0)
            
        # Chain rule: d/dx E(pts_proj(x))
        # pts_proj[i] = x[i] / ||x[i]||
        # d(pts_proj[i])/dx[i] = (I - u_i u_i^T) / ||x[i]||
        # where u_i = x[i]/||x[i]||
        
        grads = np.zeros_like(pts)
        for i in range(N):
            u = pts_proj[i]
            proj = np.eye(3) - np.outer(u, u)
            grads[i] = (proj @ grads_proj[i]) / norms_safe[i]
            
        # Gradient of penalty: 100 * sum (||x_i|| - 1)^2
        # d/dx_i = 100 * 2 * (||x_i|| - 1) * x_i / ||x_i||
        penalty_grad = 200.0 * (norms_safe - 1.0)[:, np.newaxis] * pts / norms_safe[:, np.newaxis]
        
        return (grads + penalty_grad).flatten()

    try:
        res = minimize(penalized_energy, x0, jac=penalized_grad, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-10})
        if res.success or res.fun < best_energy:
            # Project final result to sphere
            final_pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(final_pts, axis=1)
            final_pts = final_pts / norms[:, np.newaxis]
            final_energy = compute_energy_fsum(final_pts)
            if final_energy < best_energy:
                best_pts = final_pts
    except Exception as e:
        pass # Fallback to SA result if optimization fails

    return best_pts
