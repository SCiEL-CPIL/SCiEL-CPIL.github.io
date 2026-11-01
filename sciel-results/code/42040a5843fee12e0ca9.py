import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_sphere(N)
    
    def calc_energy(pts):
        """Vectorized log energy using log(d^2) for numerical stability."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-30)
        np.fill_diagonal(dist_sq, 1.0)
        # log(1/d) = -0.5 * log(d^2), sum over i<j means half the full sum
        log_d2 = np.log(dist_sq)
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
        dist_sq = np.maximum(dist_sq, 1e-30)
        np.fill_diagonal(dist_sq, 1.0)
        # Gradient of -0.25 * sum log(d^2) w.r.t. x_i:
        # d/dx_i [-0.25 * log(|x_i - x_j|^2)] = -(x_i - x_j)/(2*d^2)
        forces = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        # Project to tangent space
        dot_products = np.sum(forces * pts, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * pts
        return proj_forces

    best_energy = calc_energy(pts)
    best_pts = pts.copy()
    
    # Phase 1: Projected gradient descent with adaptive step size
    step_size = 0.05
    momentum = 0.85
    v = np.zeros_like(pts)
    max_iters = 50000
    consecutive_rejects = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 50.0:  # Leave 50s for polish phases
            break
        
        # Compute gradient
        grad = gradient(pts)
        
        # Momentum update
        v = momentum * v + (1.0 - momentum) * grad
        
        # Adaptive step size: try to accept, back off if rejected
        accepted = False
        trial_step = step_size
        for _ in range(10):
            new_pts = pts + trial_step * v
            norms = np.sqrt(np.sum(new_pts**2, axis=1))
            new_pts = new_pts / norms[:, np.newaxis]
            
            new_energy = calc_energy(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
                pts = new_pts
                accepted = True
                break
            else:
                trial_step *= 0.5
        
        if accepted:
            consecutive_rejects = 0
            # Gently increase step size
            step_size = min(step_size * 1.01, 0.15)
        else:
            consecutive_rejects += 1
            step_size *= 0.7
            v = v * 0.5
            if step_size < 1e-8 or consecutive_rejects > 50:
                break

    # Phase 2: L-BFGS-B polish with moderate tolerances
    if time.time() < deadline - 30.0:
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
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 200, 'm': 20}
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
    if time.time() < deadline - 10.0:
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
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 300, 'm': 25}
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
        
        # Phase 4: Ultra-fine final polish if time remains
        if time.time() < deadline - 3.0:
            x0 = best_pts.flatten()
            try:
                result3 = minimize(
                    objective_final, x0, jac=gradient_final,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 500, 'm': 30}
                )
                if result3.fun < best_energy:
                    res_pts = result3.x.reshape(N, 3)
                    norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-15)
                    res_pts = res_pts / norms
                    best_energy = result3.fun
                    best_pts = res_pts
            except Exception:
                pass

    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms

    return best_pts
