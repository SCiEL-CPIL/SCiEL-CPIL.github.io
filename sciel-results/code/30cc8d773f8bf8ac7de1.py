"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 150

# EVOLVE-BLOCK-START

def construct():
    """
    Minimize logarithmic energy of N points on the unit sphere.
    Strategy:
      1. Generate multiple diverse initial configurations (Fibonacci spiral, random, etc.)
      2. For each, run a multi-stage optimization:
         a. Fast gradient descent with adaptive step size
         b. L-BFGS-B from scipy (if available) for further refinement
         c. Final polish with very small steps
      3. Return the best configuration found within the time budget.
    """
    deadline = time.time() + 145.0  # 15s safety margin
    
    def log_energy(pts):
        """Compute sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|"""
        # pts: (N, 3) array on unit sphere
        n = pts.shape[0]
        # Compute all pairwise distances
        # Using broadcasting: (N, 1, 3) - (1, N, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        # Energy = sum over i<j of log(1/d) = -sum over i<j of log(d)
        # To avoid log(0), use the upper triangle
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        # Clamp to avoid log(0)
        d = np.maximum(d, 1e-15)
        return -np.sum(np.log(d))
    
    def log_energy_grad(pts):
        """
        Compute gradient of log energy with respect to pts.
        Energy E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
        dE/dxi = sum_{j!=i} (xi - xj) / |xi - xj|^2  ... wait, let me re-derive.
        
        E = sum_{i<j} -log(|xi - xj|)
        dE/dxi = sum_{j != i} d/dxi [-log|xi-xj|]
               = sum_{j != i} -(xi - xj)/|xi - xj|^2
        But we need to project this onto the tangent space of the sphere (perpendicular to xi).
        """
        n = pts.shape[0]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
        # Avoid division by zero on diagonal
        dists_safe = np.where(dists > 1e-15, dists, 1.0)
        # Gradient contribution: for each i, sum over j != i of -(xi - xj)/|xi-xj|^2
        # diff[i,j] = xi - xj
        # We want: grad_i = sum_{j != i} -(xi - xj) / |xi - xj|^2
        # = -sum_{j != i} diff[i,j] / dists[i,j]^2
        inv_d2 = np.zeros_like(dists)
        mask = ~np.eye(n, dtype=bool)
        inv_d2[mask] = 1.0 / (dists_safe[mask] ** 2)
        # grad = -sum over j of diff[i,j] * inv_d2[i,j] for each i
        # diff: (N, N, 3), inv_d2: (N, N)
        grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (N, 3)
        # Project onto tangent space: grad_tangent = grad - (grad . xi) * xi
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - dot * pts
        return grad_tangent, log_energy(pts)
    
    def project_to_sphere(pts):
        """Normalize points to lie on unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return pts / norms
    
    def optimize_from(pts0, time_limit):
        """
        Optimize starting from pts0, respecting time_limit.
        Returns (best_pts, best_energy).
        """
        best_pts = pts0.copy()
        best_energy = log_energy(pts0)
        
        # Stage 1: Gradient descent with adaptive step
        pts = pts0.copy()
        step = 0.01
        patience = 50
        no_improve = 0
        max_iter = 5000
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            grad, energy = log_energy_grad(pts)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Adaptive step: decrease if energy increased, increase if decreased
            if no_improve > patience:
                step *= 0.5
                no_improve = 0
                if step < 1e-10:
                    break
            elif no_improve == 0 and it > 100:
                step *= 1.05
            
            # Gradient descent step
            pts = pts - step * grad
            pts = project_to_sphere(pts)
        
        # Stage 2: L-BFGS-B refinement
        try:
            from scipy.optimize import minimize
            
            def objective_and_grad(x):
                pts = x.reshape(N, 3)
                pts = project_to_sphere(pts)
                grad, energy = log_energy_grad(pts)
                # Need gradient in the unconstrained space (x, not on sphere)
                # Since we project, we use the tangent gradient as approximation
                return energy, grad.flatten()
            
            x0 = best_pts.flatten()
            res = minimize(
                objective_and_grad, x0, method='L-BFGS-B',
                jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_energy:
                res_pts = project_to_sphere(res.x.reshape(N, 3))
                res_energy = log_energy(res_pts)
                if res_energy < best_energy:
                    best_energy = res_energy
                    best_pts = res_pts
        except Exception:
            pass
        
        # Stage 3: Final polish with tiny steps
        pts = best_pts.copy()
        step = 1e-5
        for it in range(2000):
            if time.time() > time_limit:
                break
            grad, energy = log_energy_grad(pts)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            pts = pts - step * grad
            pts = project_to_sphere(pts)
        
        return best_pts, best_energy
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random points
    np.random.seed(42)
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        pts = np.random.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    # 3. Fibonacci with different offset
    i = np.arange(N) + 0.25
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + 17)
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 4. Sphere with slight perturbation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts4 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    np.random.seed(789)
    pts4 = pts4 + 0.05 * np.random.randn(N, 3)
    pts4 = pts4 / np.linalg.norm(pts4, axis=1, keepdims=True)
    inits.append(pts4)
    
    best_pts = inits[0]
    best_energy = log_energy(inits[0])
    
    # Optimize each initial configuration with time budget
    n_inits = len(inits)
    time_per_init = (deadline - time.time()) / n_inits
    if time_per_init < 5.0:
        time_per_init = 5.0  # Minimum time per init
    
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        alloc = min(time_per_init, remaining - 2.0)
        if alloc < 1.0:
            break
        result_pts, result_energy = optimize_from(init_pts, time.time() + alloc)
        if result_energy < best_energy:
            best_energy = result_energy
            best_pts = result_pts
    
    # Final check: ensure points are valid
    best_pts = project_to_sphere(best_pts)
    
    # Verify no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        # This shouldn't happen, but if it does, return a valid configuration
        return inits[0]
    
    return best_pts

# EVOLVE-BLOCK-END
