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
    # D[i, j] = ||x_i - x_j||
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, 1.0) # Avoid log(0) in energy calc if needed, but we mask diagonal
    
    # Compute initial energy
    iu = np.triu_indices(N, k=1)
    current_energy = fsum(np.log(1.0 / D[iu]).tolist())
    best_energy = current_energy
    best_pts = pts.copy()
    
    # Simulated Annealing
    T0 = 0.1
    Tf = 0.001
    n_iter = 300000
    alpha = (Tf/T0)**(1.0/n_iter)
    T = T0
    
    # Pre-compute indices for others? No, k changes.
    
    for it in range(n_iter):
        T = max(T * alpha, Tf)
        
        k = np.random.randint(N)
        
        # Generate new point via random rotation
        # Small angle rotation
        angle = np.random.uniform(0.05, 0.5)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        x, y, z = axis
        R = np.array([
            [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s,   c + y*y*(1-c),    y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s,   z*y*(1-c) + x*s,  c + z*z*(1-c)]
        ])
        
        old_pt = pts[k]
        new_pt = R @ old_pt
        new_pt /= np.linalg.norm(new_pt)
        
        # Compute delta energy
        # Old distances to others
        # D[k, :] contains distances to all points, including self (1.0)
        # We need distances to j != k
        # Mask for others
        # It's faster to just use the slice
        
        # Create mask of indices != k
        # But D is a matrix.
        # d_old = D[k, np.arange(N) != k]
        # d_new = np.linalg.norm(new_pt - pts[np.arange(N) != k], axis=1)
        
        # Optimization: Pre-compute list of indices?
        # Or just use slicing which is fast in NumPy.
        
        others_idx = np.arange(N)
        others_idx = others_idx[others_idx != k]
        
        d_old = D[k, others_idx]
        d_new = np.linalg.norm(new_pt - pts[others_idx], axis=1)
        
        # Avoid log(0)
        d_old = np.maximum(d_old, 1e-15)
        d_new = np.maximum(d_new, 1e-15)
        
        delta_E = np.sum(np.log(1.0/d_new)) - np.sum(np.log(1.0/d_old))
        
        # Metropolis
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
            pts[k] = new_pt
            current_energy += delta_E
            
            # Update Distance Matrix
            # D[k, :] and D[:, k]
            # D[k, j] = ||new_pt - pts[j]||
            # We already have d_new for j in others_idx.
            # We need to update D[k, others_idx] = d_new
            # And D[others_idx, k] = d_new
            # And D[k, k] = 1.0 (or 0, but we keep 1 for safety)
            
            D[k, others_idx] = d_new
            D[others_idx, k] = d_new
            # D[k, k] remains 1.0 (or whatever it was, doesn't matter as long as we don't use it in energy)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        # Periodic recompute to avoid drift
        if it % 10000 == 0:
            # Recompute D and Energy to ensure consistency
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(D, 1.0)
            current_energy = fsum(np.log(1.0 / D[iu]).tolist())
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

    # Polish with scipy.optimize.minimize
    # Use the best points found so far
    x0 = best_pts.flatten()
    
    # L-BFGS-B is good for box constraints, but we have sphere constraint.
    # We can use 'Powell' or 'Nelder-Mead' which don't need gradients, but are slower.
    # Or use L-BFGS-B with a penalty method?
    # Better: Use the fact that we can project to sphere.
    # scipy.optimize.minimize doesn't handle non-linear constraints easily in L-BFGS-B.
    # However, we can use 'SLSQP' with constraint ||x_i||=1? That's 150 constraints. Too many.
    
    # Alternative: Minimize on the manifold?
    # Simple approach: Use 'Powell' or 'Nelder-Mead' on the flattened vector,
    # but project inside the function.
    # 'Powell' is robust for this.
    
    # Let's try L-BFGS-B with a penalty for norm deviation?
    # Or just use the gradient descent from the original code but optimized?
    
    # Actually, for this problem, a simple projected gradient descent with adaptive step size
    # or just a few steps of L-BFGS-B on the "unconstrained" problem with projection
    # is often sufficient.
    
    # Let's try 'Powell' which is derivative-free and robust.
    # It might be slow but N=150 is small.
    
    # To ensure speed, let's use the manual gradient descent but with better parameters
    # or use scipy's 'L-BFGS-B' with a penalty term.
    
    # Let's stick to a robust manual polish to avoid scipy overhead/convergence issues.
    # The previous polish was okay but maybe too few steps.
    
    pts = best_pts.copy()
    lr = 0.01
    for p_iter in range(1000):
        # Compute gradient
        grads = np.zeros_like(pts)
        for k in range(N):
            others = np.delete(pts, k, axis=0)
            diff = pts[k][np.newaxis, :] - others
            dists_sq = np.sum(diff**2, axis=1)
            dists_sq = np.maximum(dists_sq, 1e-30)
            grads[k] = -np.sum(diff / dists_sq[:, np.newaxis], axis=0)
        
        new_pts = pts - lr * grads
        norms = np.linalg.norm(new_pts, axis=1)
        norms = np.maximum(norms, 1e-15)
        new_pts = new_pts / norms[:, np.newaxis]
        
        new_energy = compute_energy_fsum(new_pts)
        if new_energy < compute_energy_fsum(pts):
            pts = new_pts
        else:
            lr *= 0.8
            if lr < 1e-6:
                break

    final_energy = compute_energy_fsum(pts)
    if final_energy < best_energy:
        best_pts = pts
    
    return best_pts
