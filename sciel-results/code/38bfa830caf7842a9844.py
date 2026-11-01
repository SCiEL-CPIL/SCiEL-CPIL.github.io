import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and Euclidean gradient for N points on the sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Vectorized pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist_ij)
    # We compute sum over all i!=j and divide by 2
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    energy = -np.sum(np.log(dist[mask])) / 2.0
    
    # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_lbfgs(pts, max_iter=1000, deadline=None):
    """
    Optimize points on the sphere using L-BFGS-B with Riemannian gradient projection.
    """
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        x = _project_to_sphere(x)
        energy, euclidean_grad = _energy_and_grad(x)
        
        # Project gradient to tangent space
        dot = np.sum(euclidean_grad * x, axis=1, keepdims=True)
        rgrad = euclidean_grad - dot * x
        
        return energy, rgrad.flatten()

    # Initial state
    energy, rgrad = objective_and_grad(pts.flatten())
    
    # L-BFGS-B expects a function and a Jacobian
    def obj_func(x):
        e, _ = objective_and_grad(x)
        return e
        
    def jac_func(x):
        _, g = objective_and_grad(x)
        return g

    # Run L-BFGS-B
    # We use the flattened coordinates. The projection in obj_func/jac_func ensures
    # we stay on the manifold effectively for the optimizer's view.
    # Note: This is a standard trick for constrained optimization where the constraint
    # is handled by projection within the objective/gradient evaluation.
    
    x0 = pts.flatten()
    
    # Check time
    if deadline and time.time() > deadline:
        return pts, energy

    try:
        result = minimize(
            obj_func, 
            x0, 
            jac=jac_func, 
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 20}
        )
        
        opt_pts = _project_to_sphere(result.x.reshape(n, 3))
        opt_energy, _ = _energy_and_grad(opt_pts)
        
        # Compare with initial
        if opt_energy < energy:
            return opt_pts, opt_energy
        else:
            return pts, energy
            
    except Exception:
        return pts, energy

def _riemannian_descent(pts, max_iters=2000, tol=1e-10, deadline=None):
    """
    Riemannian gradient descent with backtracking line search.
    """
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    energy, euclidean_grad = _energy_and_grad(pts)
    dot = np.sum(euclidean_grad * pts, axis=1, keepdims=True)
    rgrad = euclidean_grad - dot * pts
    grad_norm = np.linalg.norm(rgrad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    min_lr = 1e-6
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        if grad_norm < tol:
            break
            
        # Backtracking line search
        alpha = lr
        for _ in range(30):
            pts_trial = pts + alpha * rgrad
            pts_trial = _project_to_sphere(pts_trial)
            
            trial_energy, _ = _energy_and_grad(pts_trial)
            
            if trial_energy < energy:
                break
            alpha *= 0.5
            
        if trial_energy < energy:
            pts = pts_trial
            energy = trial_energy
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
                
            # Update gradient
            _, euclidean_grad = _energy_and_grad(pts)
            dot = np.sum(euclidean_grad * pts, axis=1, keepdims=True)
            rgrad = euclidean_grad - dot * pts
            grad_norm = np.linalg.norm(rgrad)
            
            # Increase LR if successful
            if trial_energy < energy - 1e-8:
                lr = min(lr * 1.2, 0.1)
            else:
                lr = max(lr * 0.9, min_lr)
        else:
            # If no step found, reduce LR
            lr = max(lr * 0.5, min_lr)
            if lr == min_lr:
                break

    return best_pts, best_energy

def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points with random rotation."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
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

def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _simulated_annealing(pts, t_start=0.05, t_end=0.001, max_iter=100, cooling_rate=0.98, deadline=None):
    """Refine with simulated annealing to escape local minima."""
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    energy, _ = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    t = t_start
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Generate random perturbation in tangent space
        noise = rng.standard_normal((n, 3))
        radial_comp = (noise * pts).sum(axis=1, keepdims=True)
        noise_tangent = noise - radial_comp * pts
        noise_tangent *= 0.05 * t
        
        new_pts = pts + noise_tangent
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / t):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
                
        t *= cooling_rate
        if t < t_end:
            break
            
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Multiple restarts with Riemannian L-BFGS and Descent
    # We use a mix of Fibonacci and random starts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        if restart == 0:
            pts = _fibonacci_sphere(N, seed=0)
        elif restart < 4:
            pts = _random_sphere(N, rng)
        elif restart < 7:
            pts = _fibonacci_sphere(N, seed=restart + 100)
        else:
            # Clustered starts
            centers = rng.standard_normal((5, 3))
            centers = _project_to_sphere(centers)
            pts = []
            for i in range(N):
                center = centers[i % 5]
                noise = rng.standard_normal(3) * 0.1
                pt = center + noise
                pt = pt / np.linalg.norm(pt)
                pts.append(pt)
            pts = np.array(pts)
            
        # Try L-BFGS first, then Descent if it fails or doesn't improve much
        try:
            # L-BFGS
            opt_pts, opt_energy = _riemannian_lbfgs(pts, max_iter=200, deadline=deadline)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
            # Polish with Descent if time allows
            if time.time() < deadline - 5.0:
                desc_pts, desc_energy = _riemannian_descent(opt_pts, max_iters=500, tol=1e-10, deadline=deadline)
                if desc_energy < best_energy:
                    best_energy = desc_energy
                    best_pts = desc_pts.copy()
                    
        except Exception:
            continue
            
    # Phase 2: Simulated Annealing on the best result
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            sa_pts, sa_energy = _simulated_annealing(best_pts, t_start=0.03, t_end=0.001,
                                                     max_iter=100, deadline=deadline)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
                
            # Polish SA result
            if time.time() < deadline - 5.0:
                desc_pts, desc_energy = _riemannian_descent(sa_pts, max_iters=500, tol=1e-10, deadline=deadline)
                if desc_energy < best_energy:
                    best_energy = desc_energy
                    best_pts = desc_pts.copy()
                    
        except Exception:
            pass
            
    # Phase 3: Final High-Precision Polish
    if best_pts is not None and time.time() < deadline - 2.0:
        try:
            final_pts, final_energy = _riemannian_descent(best_pts, max_iters=1000, tol=1e-12, deadline=deadline)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, seed=0)
    
    # Ensure points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += rng.standard_normal(3) * 1e-6
                best_pts[j] /= np.linalg.norm(best_pts[j])
                
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N, seed=0)
        
    return best_pts

# EVOLVE-BLOCK-END
