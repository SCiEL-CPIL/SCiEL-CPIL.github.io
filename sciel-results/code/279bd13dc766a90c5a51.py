"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    energy = -np.log(dist).sum() / 2.0
    
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    
    sum_inv_dist2 = inv_dist2.sum(axis=1)
    weighted_sum = (inv_dist2[:, :, np.newaxis] * pts[np.newaxis, :, :]).sum(axis=1)
    grad = pts * sum_inv_dist2[:, np.newaxis] - weighted_sum
    
    radial_comp = (grad * pts).sum(axis=1, keepdims=True)
    grad_tangent = grad - radial_comp * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize(pts, max_iter=500, tol=1e-10):
    pts = pts.copy()
    
    def objective_and_grad(flat_pts):
        pts_reshaped = flat_pts.reshape(-1, 3)
        pts_reshaped = _project_to_sphere(pts_reshaped)
        energy, grad = _energy_and_grad(pts_reshaped)
        radial_comp = (grad * pts_reshaped).sum(axis=1, keepdims=True)
        grad_tangent = grad - radial_comp * pts_reshaped
        return energy, grad_tangent.flatten()
    
    result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B', 
                      jac=True, options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50})
    
    opt_pts = _project_to_sphere(result.x.reshape(-1, 3))
    opt_energy, _ = _energy_and_grad(opt_pts)
    
    if result.success or result.fun < opt_energy + 1e-6:
        result2 = minimize(objective_and_grad, opt_pts.flatten(), method='L-BFGS-B',
                           jac=True, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 30})
        polish_pts = _project_to_sphere(result2.x.reshape(-1, 3))
        polish_energy, _ = _energy_and_grad(polish_pts)
        if polish_energy < opt_energy:
            opt_pts = polish_pts
            opt_energy = polish_energy
    
    return opt_pts, opt_energy

def _simulated_annealing(pts, t_start=0.1, t_end=0.001, max_iter=200, cooling_rate=0.98):
    pts = pts.copy()
    n = len(pts)
    energy, _ = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    t = t_start
    for it in range(max_iter):
        noise = np.random.randn(n, 3)
        radial_comp = (noise * pts).sum(axis=1, keepdims=True)
        noise_tangent = noise - radial_comp * pts
        noise_tangent *= 0.05 * t
        
        new_pts = pts + noise_tangent
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - energy
        
        if delta_e < 0 or (t > 1e-6 and np.random.rand() < math.exp(min(0, -delta_e / t))):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        t *= cooling_rate
        if t < t_end:
            break
    
    return best_pts, best_energy

def _fibonacci_sphere(n, seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    theta = rng.uniform(0, 2 * np.pi)
    phi_rot = rng.uniform(-np.pi/2, np.pi/2)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi_rot), np.sin(phi_rot)
    
    R = np.array([
        [cos_t, sin_t, 0],
        [-sin_t, cos_t, 0],
        [0, 0, 1]
    ])
    R2 = np.array([
        [1, 0, 0],
        [0, cos_p, sin_p],
        [0, -sin_p, cos_p]
    ])
    R_total = R2 @ R
    pts = pts @ R_total.T
    
    return pts

def _riemannian_descent(pts, lr=0.01, max_iter=2000, tol=1e-12):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    pts = pts.copy()
    n = len(pts)
    energy, grad = _energy_and_grad(pts)
    
    for it in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        
        # Tangent step
        step = -lr * grad
        # Exponential map (approximation for small steps)
        new_pts = pts + step
        new_pts = _project_to_sphere(new_pts)
        
        # Backtracking line search
        alpha = 1.0
        for _ in range(20):
            trial_pts = _project_to_sphere(pts + alpha * step)
            trial_energy, _ = _energy_and_grad(trial_pts)
            if trial_energy < energy - 1e-4 * alpha * np.linalg.norm(grad)**2:
                pts = trial_pts
                energy = trial_energy
                break
            alpha *= 0.5
        
        if alpha < 1e-6:
            break
    
    return pts, energy

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Multiple restarts with L-BFGS-B
    for restart in range(15):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = _fibonacci_sphere(N, seed=restart)
        elif restart < 5:
            rng = np.random.RandomState(restart * 100 + 42)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart < 10:
            pts = _fibonacci_sphere(N, seed=restart + 100)
        else:
            rng = np.random.RandomState(restart * 200 + 7)
            centers = rng.randn(5, 3)
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            pts = []
            for i in range(N):
                center = centers[i % 5]
                noise = rng.randn(3) * 0.1
                pt = center + noise
                pt = pt / np.linalg.norm(pt)
                pts.append(pt)
            pts = np.array(pts)
        
        try:
            opt_pts, opt_energy = _optimize(pts, max_iter=400)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Phase 2: Simulated annealing + re-optimization on best
    if best_pts is not None and time.time() < deadline - 15:
        try:
            sa_pts, sa_energy = _simulated_annealing(best_pts, t_start=0.08, t_end=0.0005, max_iter=150)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
            
            if time.time() < deadline - 10:
                re_opt_pts, re_opt_energy = _optimize(sa_pts, max_iter=200)
                if re_opt_energy < best_energy:
                    best_energy = re_opt_energy
                    best_pts = re_opt_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Riemannian descent polish
    if best_pts is not None and time.time() < deadline - 10:
        try:
            rd_pts, rd_energy = _riemannian_descent(best_pts, lr=0.005, max_iter=1000)
            if rd_energy < best_energy:
                best_energy = rd_energy
                best_pts = rd_pts.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if best_pts is not None and time.time() < deadline - 5:
        try:
            final_pts, final_energy = _optimize(best_pts, max_iter=300)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, seed=0)
    
    return best_pts
# EVOLVE-BLOCK-END
