"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())

def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient w.r.t. free R^3 coordinates.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero
    inv_dist_sq = np.zeros_like(dist_sq)
    off_diag = dist_sq > 1e-30
    inv_dist_sq[off_diag] = 1.0 / dist_sq[off_diag]
    
    # Gradient of E = -sum log(d) is -sum (x_i - x_j)/d^2
    # diff[i,j] = x_i - x_j
    # grad_i = - sum_j diff[i,j] / dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _project_to_tangent(pts, grad):
    """Project gradient onto tangent space of unit sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

def _make_fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _random_sphere(n):
    """Generate n random points on the unit sphere."""
    pts = np.random.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _adam_optimize(pts0, time_limit=100.0):
    """Optimize points on the sphere using Adam gradient descent with adaptive learning rate."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Adam parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Initialize moments
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    start_time = time.time()
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _log_energy_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        if no_improve_steps > max_no_improve:
            break
            
        # Project gradient to tangent space
        grad_tangent = _project_to_tangent(pts, grad)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent * grad_tangent)
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update with step size clipping
        step = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        # Clip step size to prevent divergence
        step_norms = np.linalg.norm(step, axis=1, keepdims=True)
        max_step = 0.15
        scale = np.minimum(1.0, max_step / (step_norms + 1e-15))
        step = step * scale
        
        pts_new = pts - step
        
        # Retract to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check for convergence or divergence
        if not np.all(np.isfinite(pts_new)):
            break
            
        pts = pts_new
        
        # Adaptive learning rate: decay then periodically reset for escape
        if t % 200 == 0:
            lr *= 0.95
            if lr < 1e-8:
                # Reset momentum and boost lr to escape local minima
                lr = 0.005
                m = np.zeros_like(pts)
                v = np.zeros_like(pts)
                no_improve_steps = 0
                if time.time() - start_time > time_limit * 0.8:
                    break

    return best_pts, best_energy

def _lbfgs_polish(pts0, time_limit=10.0):
    """Polish solution using L-BFGS-B with line search. Multiple passes with tightening tolerances."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    start_time = time.time()
    
    # Flatten to 1D for scipy
    def objective(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        e, _ = _log_energy_grad(pts_norm)
        return e
    
    def gradient(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        _, g = _log_energy_grad(pts_norm)
        # Project to tangent and flatten
        g_tan = _project_to_tangent(pts_norm, g)
        return g_tan.flatten()
    
    # Multiple polish passes with progressively tighter tolerances
    passes = [
        {'maxiter': 300, 'maxfun': 600, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
        {'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
        {'maxiter': 800, 'maxfun': 1500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50},
    ]
    
    x0 = pts.flatten()
    
    for opt in passes:
        if time.time() - start_time > time_limit:
            break
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options=opt
            )
            
            if result.fun < best_energy:
                pts_opt = result.x.reshape(-1, 3)
                norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
                pts_opt = pts_opt / norms
                e_opt = _log_energy(pts_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = pts_opt
                    x0 = pts_opt.flatten()
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 145.0  # 5 second safety margin
    
    # Start with Fibonacci spiral
    pts0 = _make_fibonacci_sphere()
    
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Main optimization from Fibonacci start - give it substantial time
    remaining = deadline - time.time()
    if remaining > 30.0:
        pts_opt, energy_opt = _adam_optimize(pts0, time_limit=remaining - 40.0)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Cluster-based restarts: perturb subsets of the best solution
    np.random.seed(42)
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Create cluster-based perturbation: move a random subset of points
        n_cluster = 10 + 5 * (r % 4)  # 10, 15, 20, 25, 10, 15, 20, 25
        cluster_idx = np.random.choice(N, size=n_cluster, replace=False)
        pts_rand = best_pts.copy()
        # Perturb cluster points
        perturb_scale = 0.05 + 0.03 * (r % 3)
        pts_rand[cluster_idx] += perturb_scale * np.random.randn(n_cluster, 3)
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
            
        # Time allocation: more time for earlier restarts
        t_limit = max(remaining * 0.35, 3.0)
        pts_opt, energy_opt = _adam_optimize(pts_rand, time_limit=t_limit)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Final fine-tuning with Adam on best solution
    if time.time() < deadline - 15.0:
        remaining = max(deadline - time.time() - 10.0, 2.0)
        pts_opt, energy_opt = _adam_optimize(best_pts, time_limit=remaining)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # L-BFGS polishing - give it good time for high precision with multiple passes
    if time.time() < deadline - 5.0:
        remaining = max(deadline - time.time() - 3.0, 3.0)
        pts_opt, energy_opt = _lbfgs_polish(best_pts, time_limit=remaining)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Ensure points are on the unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Check for very close points and perturb if necessary
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dist[iu]) < 1e-9:
        for _ in range(100):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            rand_dir = np.random.randn(3)
            rand_dir /= np.linalg.norm(rand_dir)
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break
    
    return best_pts

# EVOLVE-BLOCK-END
