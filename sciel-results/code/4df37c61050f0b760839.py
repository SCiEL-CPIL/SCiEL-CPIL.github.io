"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, Bounds

def _energy_fast(pts):
    """Vectorized energy computation using math.fsum for precision."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    # Only use upper triangle
    iu = np.triu_indices(n, 1)
    d_sq = dists_sq[iu]
    d = np.sqrt(d_sq)
    d = np.maximum(d, 1e-15)
    # Use math.fsum for better precision as specified in task
    return math.fsum(math.log(1.0 / di) for di in d)

def _energy_and_grad_vectorized(pts):
    """Compute energy and gradient efficiently using vectorized operations."""
    n = len(pts)
    # Pairwise differences: shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Squared distances
    dists_sq = np.sum(diff * diff, axis=2)
    # Distances
    dists = np.sqrt(dists_sq)
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Energy: sum of log(1/d) = -sum of log(d) for i<j
    d_upper = dists[mask]
    d_upper = np.maximum(d_upper, 1e-15)
    energy = math.fsum(math.log(1.0 / di) for di in d_upper)
    
    # Safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists_sq = np.where(dists_sq > 1e-30, 1.0 / dists_sq, 0.0)
    
    # grad_i = -sum_j (xi-xj)/|xi-xj|^2
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    grad_proj = grad.copy()
    for i in range(n):
        dot = np.dot(grad[i], pts[i])
        grad_proj[i] = grad[i] - dot * pts[i]
    
    return energy, grad_proj

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(rng):
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)

def _tangent_optimize(pts, max_time=50.0, gtol=1e-8):
    """Optimize on the sphere using projected gradient with line search."""
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    current_energy, current_grad = _energy_and_grad_vectorized(pts)
    best_energy = current_energy
    best_pts = pts.copy()
    
    # Conjugate gradient method on the sphere
    prev_grad = None
    prev_dir = None
    
    max_iter = 5000
    for it in range(max_iter):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > max_time - 1.0:
            break
        
        # Compute gradient
        energy, grad = _energy_and_grad_vectorized(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < gtol:
            break
        
        # Conjugate gradient direction
        if prev_grad is not None:
            # Polak-Ribiere beta
            num = np.sum(grad * (grad - prev_grad))
            den = np.sum(prev_grad * prev_grad)
            beta = max(0.0, num / den) if den > 1e-30 else 0.0
            direction = -grad + beta * prev_dir
        else:
            direction = -grad
            beta = 0.0
        
        # Project direction to tangent space
        dir_proj = direction.copy()
        for i in range(n):
            dot = np.dot(direction[i], pts[i])
            dir_proj[i] = direction[i] - dot * pts[i]
        
        # Normalize direction for line search
        dir_norm = np.linalg.norm(dir_proj)
        if dir_norm < 1e-15:
            break
        dir_proj = dir_proj / dir_norm
        
        # Backtracking line search
        alpha = 0.1
        c1 = 1e-4
        for ls in range(30):
            # Move in tangent direction and project back to sphere
            new_pts = pts + alpha * dir_proj
            new_pts = _project_to_sphere(new_pts)
            
            new_energy, _ = _energy_and_grad_vectorized(new_pts)
            
            if new_energy < energy - c1 * alpha * np.sum(grad * dir_proj):
                # Accept step
                pts = new_pts
                break
            alpha *= 0.5
        else:
            # If no step was accepted, try a smaller step or break
            break
        
        prev_grad = grad
        prev_dir = dir_proj
    
    return best_pts, best_energy

def _lbfgs_sphere(pts, max_time=30.0, maxiter=300, ftol=1e-8, gtol=1e-8):
    """Use L-BFGS-B with sphere projection."""
    start_time = time.time()
    pts = pts.copy()
    
    def obj_grad(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad_vectorized(p)
        return e, g.reshape(-1)
    
    best_energy, _ = _energy_and_grad_vectorized(pts)
    best_pts = pts.copy()
    
    try:
        res = minimize(
            obj_grad,
            pts.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxfun': maxiter}
        )
        if time.time() - start_time < max_time:
            new_pts = _project_to_sphere(res.x.reshape(N, 3))
            new_energy, _ = _energy_and_grad_vectorized(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_energy

def _high_precision_polish(pts, max_time=20.0):
    """Final high-precision polish using tight tolerances."""
    start_time = time.time()
    pts = pts.copy()
    
    # First do a very tight L-BFGS-B
    best_pts, best_energy = _lbfgs_sphere(pts, max_time=max_time * 0.6, maxiter=500, ftol=1e-15, gtol=1e-14)
    
    remaining = max_time - (time.time() - start_time)
    if remaining > 2:
        # Then do a tight CG polish
        cg_pts, cg_energy = _tangent_optimize(best_pts, max_time=remaining, gtol=1e-12)
        if cg_energy < best_energy:
            best_energy = cg_energy
            best_pts = cg_pts
    
    return best_pts, best_energy

def _optimize_from_start(pts, max_time=60.0):
    """Multi-method optimization from a starting configuration."""
    start_time = time.time()
    current_energy, _ = _energy_and_grad_vectorized(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    remaining = max_time - (time.time() - start_time)
    if remaining < 3:
        return best_pts, best_energy
    
    # Method 1: Tangent-space conjugate gradient
    time_budget1 = min(remaining * 0.35, 25.0)
    if time_budget1 > 3:
        cg_pts, cg_energy = _tangent_optimize(pts, max_time=time_budget1)
        if cg_energy < best_energy:
            best_energy = cg_energy
            best_pts = cg_pts
    
    remaining = max_time - (time.time() - start_time)
    if remaining < 3:
        return best_pts, best_energy
    
    # Method 2: L-BFGS-B
    time_budget2 = min(remaining * 0.35, 20.0)
    if time_budget2 > 3:
        lb_pts, lb_energy = _lbfgs_sphere(best_pts, max_time=time_budget2)
        if lb_energy < best_energy:
            best_energy = lb_energy
            best_pts = lb_pts
    
    remaining = max_time - (time.time() - start_time)
    if remaining < 3:
        return best_pts, best_energy
    
    # Method 3: High-precision polish
    time_budget3 = min(remaining * 0.4, 20.0)
    if time_budget3 > 3:
        hp_pts, hp_energy = _high_precision_polish(best_pts, max_time=time_budget3)
        if hp_energy < best_energy:
            best_energy = hp_energy
            best_pts = hp_pts
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral - good initial configuration
    pts0 = _fibonacci_spiral()
    
    best_pts = pts0.copy()
    best_energy, _ = _energy_and_grad_vectorized(pts0)
    
    # Optimize from Fibonacci start with generous budget
    remaining = deadline - time.time()
    if remaining > 10:
        opt_pts, opt_energy = _optimize_from_start(pts0, max_time=min(remaining - 5, 70))
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts
    
    # Try multiple random restarts if time permits
    rng = np.random.default_rng(42)
    n_restarts = 5
    for k in range(n_restarts):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        pts_r = _random_points(rng)
        # Quick local optimization
        time_budget = min(remaining - 3, 20)
        if time_budget < 5:
            break
        opt_pts_r, opt_energy_r = _optimize_from_start(pts_r, max_time=time_budget)
        if opt_energy_r < best_energy:
            best_energy = opt_energy_r
            best_pts = opt_pts_r
    
    # Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 5:
        hp_pts, hp_energy = _high_precision_polish(best_pts, max_time=min(remaining - 2, 15))
        if hp_energy < best_energy:
            best_energy = hp_energy
            best_pts = hp_pts
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-9:
                best_pts[j] += 1e-6 * rng.standard_normal(3)
                best_pts = _project_to_sphere(best_pts)
                break
    
    return best_pts
# EVOLVE-BLOCK-END
