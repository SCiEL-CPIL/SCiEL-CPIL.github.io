import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _halton_sphere(n):
    """Generate points using Halton sequence projected to sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map to sphere: use spherical coordinates
        theta = 2.0 * math.pi * x
        phi = math.acos(2.0 * y - 1.0)
        px = math.sin(phi) * math.cos(theta)
        py = math.sin(phi) * math.sin(theta)
        pz = math.cos(phi)
        points.append([px, py, pz])
    
    pts = np.array(points)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin for 150s limit

    # Initial configuration: Halton sequence on sphere
    pts = _halton_sphere(N)
    
    def calc_energy(pts):
        """Vectorized log energy computation."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        # log(1/d) = -0.5 * log(d^2)
        log_d2 = np.log(dist_sq)
        np.fill_diagonal(log_d2, 0.0)
        energy = -0.25 * np.sum(log_d2)
        return energy

    def calc_energy_fsum(pts):
        """High-precision energy using math.fsum, matching evaluator."""
        terms = []
        for i in range(N):
            for j in range(i + 1, N):
                d_sq = np.sum((pts[i] - pts[j])**2)
                if d_sq < 1e-30:
                    return float('inf')
                terms.append(math.log(1.0 / math.sqrt(d_sq)))
        return math.fsum(terms)

    def gradient(pts):
        """Compute gradient of log energy on sphere (projected to tangent)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        np.fill_diagonal(dist_sq, 1.0)
        # Gradient of -0.25 * sum log(d2) w.r.t. x_i
        # d/dx_i [-0.25 * log(|x_i - x_j|^2)] = -0.25 * 2*(x_i - x_j)/|x_i - x_j|^2 = -(x_i-x_j)/(2*d2)
        # Sum over j != i
        forces = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        # Project to tangent space
        dot_products = np.sum(forces * pts, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * pts
        return proj_forces

    best_energy = calc_energy(pts)
    best_pts = pts.copy()
    
    # Phase 1: Projected gradient descent with momentum
    step_size = 0.02
    momentum = 0.9
    v = np.zeros_like(pts)
    max_iters = 30000
    
    for it in range(max_iters):
        if time.time() > deadline - 35.0:  # Leave 35s for polish
            break
            
        # Compute gradient
        grad = gradient(pts)
        
        # Momentum update
        v = momentum * v + (1 - momentum) * grad
        grad_mom = v
        
        # Update points
        new_pts = pts + step_size * grad_mom
        norms = np.sqrt(np.sum(new_pts**2, axis=1))
        new_pts = new_pts / norms[:, np.newaxis]
        
        # Calculate new energy
        new_energy = calc_energy(new_pts)
        
        # Accept if better
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
            # Slowly increase step size if improving
            step_size = min(step_size * 1.005, 0.08)
        else:
            # Revert and reduce step size
            pts = best_pts.copy()
            step_size *= 0.9
            v = v * 0.5  # Dampen momentum
            if step_size < 1e-7:
                break

    # Phase 2: L-BFGS-B polish
    if time.time() < deadline - 15.0:
        x0 = best_pts.flatten()
        
        def objective(x):
            p = x.reshape(N, 3)
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
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < best_energy:
                res_pts = result.x.reshape(N, 3)
                norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                res_pts = res_pts / norms
                best_energy = result.fun
                best_pts = res_pts
        except Exception:
            pass

    # Phase 3: Final high-precision polish with very tight tolerances
    if time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        
        def objective_final(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return calc_energy(p)
        
        def gradient_final(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return gradient(p).flatten()
        
        try:
            result2 = minimize(
                objective_final, x0, jac=gradient_final,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
            )
            if result2.fun < best_energy:
                res_pts = result2.x.reshape(N, 3)
                norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                res_pts = res_pts / norms
                best_energy = result2.fun
                best_pts = res_pts
        except Exception:
            pass

    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms

    return best_pts
