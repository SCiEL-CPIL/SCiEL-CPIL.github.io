import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # Safety margin from 150s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    # Note: log(1/d) = -log(d) = -0.5 log(d_sq)
    # Sum over i<j is half the sum over i!=j
    # So sum_{i<j} -0.5 log(d_sq_ij) = -0.25 sum_{i!=j} log(d_sq_ij)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _grad_log_energy(pts):
    """
    Compute the Euclidean gradient of the log energy.
    dE/dx_i = -0.5 sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Weights: -0.5 / dist_sq
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # grad[i, k] = sum_j weights[i, j] * diff[i, j, k]
    grad = np.einsum('ij,ijk->ik', weights, diff)
    return grad

def _tangent_project(grad, pts):
    """
    Project Euclidean gradient to the tangent space of the sphere.
    grad_tangent = grad - (grad . x) x
    """
    proj_factor = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - proj_factor * pts

def _exp_map(pts, tangent_vecs, step_size):
    """
    Exponential map on the sphere.
    x_new = x cos(theta) + v sin(theta) / ||v||
    where theta = step_size * ||v|| (approx) or simply use rotation.
    Actually, for small steps, x + v is a good approximation, but for SA we need robust moves.
    Let's use the exact exponential map for stability.
    """
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    
    theta = step_size * norms
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Handle very small norms to avoid numerical issues with sin(theta)/norm
    # If norm is small, sin(theta)/norm ~ step_size
    # So term is step_size * tangent_vec
    v_scaled = np.where(norms > 1e-12, sin_t * tangent_vecs / safe_norms, step_size * tangent_vecs)
    
    return pts * cos_t + v_scaled

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1.0) - 1.0)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _local_refine(pts, time_limit):
    """
    Use L-BFGS-B to polish the current configuration.
    """
    start = time.time()
    try:
        def obj_and_grad(x_flat):
            p = x_flat.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            
            e = _log_energy(p)
            if not np.isfinite(e):
                return e, np.zeros_like(x_flat)
            
            g_eucl = _grad_log_energy(p)
            g_riem = _tangent_project(g_eucl, p)
            return e, g_riem.flatten()
        
        res = minimize(
            obj_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        opt_pts = res.x.reshape(N, 3)
        norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        opt_pts = opt_pts / norms
        return opt_pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    
    # Initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2)
    ]
    
    best_energy = float('inf')
    best_pts = None
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 1e-6
    cooling_rate = 0.95
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() - start_time > DEADLINE:
            break
            
        current_pts = init_pts.copy()
        current_energy = _log_energy(current_pts)
        
        if not np.isfinite(current_energy):
            continue
            
        T = T_start
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - start_time < DEADLINE - 5:
            # Compute gradient
            g_eucl = _grad_log_energy(current_pts)
            g_riem = _tangent_project(g_eucl, current_pts)
            
            # Step size: adaptive based on gradient norm? 
            # Let's use a fixed step size scaled by temperature for exploration
            # Or just use a fixed small step for gradient descent part, and random kicks for SA.
            
            # Move 1: Gradient step
            step_size = 0.01
            grad_step = -step_size * g_riem
            new_pts_grad = _exp_map(current_pts, grad_step, 1.0)
            # Normalize again for safety
            norms = np.linalg.norm(new_pts_grad, axis=1, keepdims=True)
            new_pts_grad = new_pts_grad / norms
            
            e_grad = _log_energy(new_pts_grad)
            
            # Move 2: Random perturbation (SA kick)
            # Scale perturbation by T
            pert_scale = 0.1 * T
            rng = np.random.default_rng(int(time.time()*1000) + init_idx)
            random_vecs = rng.normal(0, 1, (N, 3))
            # Project random vectors to tangent space
            proj_rand = random_vecs - np.sum(random_vecs * current_pts, axis=1, keepdims=True) * current_pts
            pert_step = pert_scale * proj_rand
            new_pts_rand = _exp_map(current_pts, pert_step, 1.0)
            norms = np.linalg.norm(new_pts_rand, axis=1, keepdims=True)
            new_pts_rand = new_pts_rand / norms
            e_rand = _log_energy(new_pts_rand)
            
            # Choose the better move or accept based on Metropolis criterion
            # For simplicity, let's try the gradient step first.
            # If gradient step improves, take it.
            # Else, check Metropolis for random step.
            
            accepted = False
            if e_grad < current_energy:
                current_pts = new_pts_grad
                current_energy = e_grad
                accepted = True
            else:
                # Check random step
                if e_rand < current_energy:
                    current_pts = new_pts_rand
                    current_energy = e_rand
                    accepted = True
                else:
                    # Metropolis acceptance for random step
                    delta_e = e_rand - current_energy
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        current_pts = new_pts_rand
                        current_energy = e_rand
                        accepted = True
            
            if accepted:
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Cool down
            T *= cooling_rate
            
            # If stuck, restart with a small perturbation or break to try next init
            if no_improve_count > max_no_improve:
                # Try a local refinement to settle
                current_pts = _local_refine(current_pts, 2.0)
                current_energy = _log_energy(current_pts)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = current_pts.copy()
                break
                
            # If T is too low, break
            if T < T_end:
                break

    # Final Polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        best_pts = _local_refine(best_pts, DEADLINE - 2 - (time.time() - start_time))
        final_energy = _log_energy(best_pts)
        if final_energy < best_energy:
            best_energy = final_energy

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
