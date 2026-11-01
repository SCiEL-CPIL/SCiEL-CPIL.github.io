import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135.0  # 135s safety margin for 150s limit

    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def calc_energy(pts):
        """Vectorized log energy computation."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        dist = np.maximum(dist, 1e-12)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = np.sum(np.log(1.0 / dist[mask]))
        return energy

    def calc_energy_fsum(pts):
        """High-precision energy using math.fsum, matching evaluator."""
        terms = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-15:
                    return float('inf')
                terms.append(math.log(1.0 / d))
        return math.fsum(terms)

    def gradient(pts):
        """Compute gradient of log energy on sphere (projected to tangent)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        # Force: sum_j (pts[i] - pts[j]) / dist_sq[i,j]
        forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        # Project to tangent space
        dot_products = np.sum(forces * pts, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * pts
        return proj_forces

    best_energy = calc_energy(pts)
    best_pts = pts.copy()
    
    # Phase 1: Projected gradient descent with adaptive step size
    step_size = 0.01
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() > deadline - 30.0:  # Leave 30s for polish
            break
            
        # Compute forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        dot_products = np.sum(forces * pts, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * pts
        
        # Update points
        new_pts = pts + step_size * proj_forces
        norms = np.sqrt(np.sum(new_pts**2, axis=1))
        new_pts = new_pts / norms[:, np.newaxis]
        
        # Calculate new energy
        new_energy = calc_energy(new_pts)
        
        # Accept if better
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
            # Slightly increase step size if improving
            step_size = min(step_size * 1.01, 0.05)
        else:
            # Revert and reduce step size
            step_size *= 0.85
            if step_size < 1e-6:
                break

    # Phase 2: High-precision polish with scipy L-BFGS-B
    if time.time() < deadline - 15.0:
        # Flatten points for scipy
        x0 = best_pts.flatten()
        
        def objective(x):
            p = x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return calc_energy(p)
        
        def gradient_func(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return gradient(p).flatten()
        
        try:
            result = minimize(
                objective, x0, jac=gradient_func,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < best_energy:
                # Normalize result
                res_pts = result.x.reshape(N, 3)
                norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                res_pts = res_pts / norms
                best_energy = result.fun
                best_pts = res_pts
        except Exception:
            pass

    # Final verification with fsum
    final_energy = calc_energy_fsum(best_pts)
    if math.isfinite(final_energy) and final_energy < best_energy:
        best_energy = final_energy

    return best_pts
