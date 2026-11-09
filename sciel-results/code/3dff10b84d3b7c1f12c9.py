"""Rewritten program using L-BFGS-B with manifold gradient handling and multi-restart."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere.
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Clip to avoid log(0) and numerical issues
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    log_dist_sq = np.log(dist_sq)
    # Set diagonal to 0 to ignore self-interaction
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient of E with respect to x_i:
    # E = -0.5 * sum_{i!=j} log(|x_i - x_j|)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / (2 * |x_i - x_j|^2)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    term1 = inv_dist_sq @ pts
    term2 = pts * np.sum(inv_dist_sq, axis=1)[:, None]
    
    grad_raw = 0.5 * (term1 - term2)
    
    # Project to tangent space
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _energy_and_grad_manifold(x_flat):
    """
    Objective function and gradient for scipy.optimize.minimize.
    x_flat: (3N,) vector.
    """
    x = x_flat.reshape(N, 3)
    x = _project_to_sphere(x)
    
    energy, grad_tan = _energy_and_tangent_grad(x)
    
    # The gradient of f(x) = E(x/||x||) with respect to x is the tangent gradient of E at u = x/||x||.
    # This is because the Jacobian of the normalization map is a projection onto the tangent space.
    return energy, grad_tan.flatten()

def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _conjugate_gradient(pts, max_iter=1000, tol=1e-10, deadline=None):
    """
    Conjugate Gradient method on the sphere with Wolfe line search.
    """
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    p = -grad
    
    # Project p to tangent space
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Line search
        alpha = 0.1
        c1 = 1e-4
        c2 = 0.9
        rho = 0.5
        
        dir_deriv = np.sum(grad * p)
        
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)
        
        found_step = False
        # Try to find a step satisfying Wolfe conditions
        # First, check if we need to expand alpha
        alpha_max = 1.0
        alpha_min = 1e-10
        
        x_new = x + alpha * p
        x_new = _project_to_sphere(x_new)
        e_new, g_new = _energy_and_tangent_grad(x_new)
        
        # Check Armijo
        if e_new <= energy + c1 * alpha * dir_deriv:
            # Check Curvature
            if np.sum(g_new * p) >= c2 * dir_deriv:
                found_step = True
            else:
                # Try smaller steps
                alpha_trial = alpha * 0.5
                for _ in range(50):
                    x_trial = x + alpha_trial * p
                    x_trial = _project_to_sphere(x_trial)
                    e_trial, g_trial = _energy_and_tangent_grad(x_trial)
                    
                    if e_trial <= energy + c1 * alpha_trial * dir_deriv and np.sum(g_trial * p) >= c2 * dir_deriv:
                        x_new = x_trial
                        e_new = e_trial
                        g_new = g_trial
                        alpha = alpha_trial
                        found_step = True
                        break
                    alpha_trial *= 0.5
                    if alpha_trial < alpha_min:
                        break
        else:
            # Try smaller steps
            alpha_trial = alpha * 0.5
            for _ in range(50):
                x_trial = x + alpha_trial * p
                x_trial = _project_to_sphere(x_trial)
                e_trial, g_trial = _energy_and_tangent_grad(x_trial)
                
                if e_trial <= energy + c1 * alpha_trial * dir_deriv:
                    if np.sum(g_trial * p) >= c2 * dir_deriv:
                        x_new = x_trial
                        e_new = e_trial
                        g_new = g_trial
                        alpha = alpha_trial
                        found_step = True
                        break
                alpha_trial *= 0.5
                if alpha_trial < alpha_min:
                    break
        
        if not found_step:
            # Fallback to best effort or break
            break
        
        x = x_new
        energy = e_new
        grad = g_new
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Polak-Ribiere
        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        
        if grad_prev_dot < 1e-16:
            beta = 0.0
        else:
            beta = max(0.0, grad_dot / grad_prev_dot)
            
        p_new = -grad + beta * p
        
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        
        if np.sum(grad * p_new) > 0:
            beta = 0.0
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
            
        p = p_new
        grad_prev = grad.copy()
        
    return x

def _lbfgs_polish(pts, deadline):
    """
    High-precision polish using L-BFGS-B.
    """
    x0 = pts.flatten()
    
    def fun(x):
        return _energy_and_grad_manifold(x)[0]
    
    def jac(x):
        return _energy_and_grad_manifold(x)[1]
    
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        x_opt = res.x.reshape(N, 3)
        x_opt = _project_to_sphere(x_opt)
        e_opt, _ = _energy_and_tangent_grad(x_opt)
        return x_opt, e_opt
    except Exception:
        return pts, _energy_and_tangent_grad(pts)[0]

def _simulate_annealing(pts, deadline, rng):
    """
    Simulated Annealing on the sphere.
    """
    x = pts.copy()
    energy, _ = _energy_and_tangent_grad(x)
    
    start_time = time.time()
    T = 0.1
    T_min = 1e-6
    cooling = 0.95
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Propose a random move
        idx = rng.integers(0, N)
        angle = rng.uniform(0, 2 * np.pi)
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        
        # Rotate point idx by angle around axis
        # Rodrigues' rotation formula
        c = np.cos(angle)
        s = np.sin(angle)
        k = axis
        xk = np.cross(k, x[idx])
        kxk = np.dot(k, k) * x[idx]
        
        new_point = x[idx] * c + xk * s + k * np.dot(k, x[idx]) * (1 - c)
        
        x_old = x[idx].copy()
        x[idx] = new_point
        
        new_energy, _ = _energy_and_tangent_grad(x)
        delta_e = new_energy - energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            energy = new_energy
        else:
            x[idx] = x_old
        
        T *= cooling
    
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_initial_fibonacci())
    
    for i in range(5):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    for i in range(5):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.1
        inits.append(_project_to_sphere(pts))
        
    best_pts = inits[0].copy()
    best_energy = _energy_and_tangent_grad(best_pts)[0]
    
    # Phase 1: Optimize each initialization with CG
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        time_left = deadline - time.time()
        run_deadline = time.time() + min(5.0, time_left - 2.0)
        
        if run_deadline <= time.time():
            continue
        
        optimized_pts = _conjugate_gradient(init_pts, max_iter=500, deadline=run_deadline)
        
        e_opt, _ = _energy_and_tangent_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
            
    # Phase 2: Simulated Annealing on the best point found so far
    if time.time() < deadline - 10:
        sa_deadline = deadline - 5
        sa_pts = _simulate_annealing(best_pts, sa_deadline, rng)
        e_sa, _ = _energy_and_tangent_grad(sa_pts)
        
        if e_sa < best_energy:
            best_energy = e_sa
            best_pts = sa_pts.copy()
            
    # Phase 3: Final L-BFGS-B polish
    if time.time() < deadline - 5:
        final_pts, e_final = _lbfgs_polish(best_pts, deadline)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
            
    # Phase 4: One last CG pass to clean up any numerical drift
    if time.time() < deadline - 1:
        final_pts = _conjugate_gradient(best_pts, max_iter=500, deadline=deadline - 0.5)
        e_final, _ = _energy_and_tangent_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()

    best_pts = _project_to_sphere(best_pts)
    
    # Safety check
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
