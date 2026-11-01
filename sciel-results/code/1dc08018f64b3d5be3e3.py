"""Minimal logarithmic energy of 150 points on the sphere - dual-representation optimizer."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_MARGIN = 5.0  # seconds before 150s wall-clock limit

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on sphere.
    pts: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Gradient of energy w.r.t. x_i (on R^3, projected to tangent space):
      dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = N
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
    
    dists = np.sqrt(dists_sq)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    
    # Check for invalid distances
    if np.any(d_upper < 1e-12):
        return float('inf'), np.zeros((n, 3))
    
    # Compute energy using fsum for precision
    terms = [math.log(1.0 / d) for d in d_upper]
    energy = math.fsum(terms)
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j inv_d2[i,j] * (x_i - x_j)
    # = x_i * sum_j inv_d2[i,j] - sum_j inv_d2[i,j] * x_j
    weighted_sum = inv_d2 @ pts  # (n, 3)
    sum_inv = np.sum(inv_d2, axis=1)  # (n,)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum  # (n, 3)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _exp_map(pts, v):
    """
    Exponential map on sphere: move points by tangent vectors v.
    pts: (N, 3) points on sphere
    v: (N, 3) tangent vectors
    Returns new points on sphere.
    """
    # For each point, v should be tangent (orthogonal to pts[i])
    # Project v to tangent space
    dot = np.sum(pts * v, axis=1, keepdims=True)
    v_tangent = v - dot * pts
    
    # Exponential map: x_new = x*cos(|v|) + (v/|v|)*sin(|v|)
    v_norms = np.linalg.norm(v_tangent, axis=1, keepdims=True)
    v_norms_safe = np.where(v_norms < 1e-15, 1.0, v_norms)
    
    cos_term = np.cos(v_norms_safe)
    sin_term = np.sin(v_norms_safe) / v_norms_safe
    sin_term = np.where(v_norms < 1e-15, 0.0, sin_term)
    
    new_pts = pts * cos_term + v_tangent * sin_term
    return _project_to_sphere(new_pts)

def _retract(pts, v, step):
    """
    Retraction on sphere: project pts + step*v back to sphere.
    """
    new_pts = pts + step * v
    return _project_to_sphere(new_pts)

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)

def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence in 3D."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)

def _lbfgs_sphere(pts, time_limit, maxiter=500, tol=1e-10):
    """
    L-BFGS optimization on sphere using penalty method.
    We optimize in R^3N with a penalty for deviation from sphere.
    """
    start_time = time.time()
    
    def obj_grad(flat):
        if time.time() - start_time > time_limit:
            return float('inf'), np.zeros_like(flat)
        
        p = flat.reshape(N, 3)
        # Project to sphere for energy/gradient computation
        p_proj = _project_to_sphere(p)
        
        energy, grad_manifold = _energy_and_grad(p_proj)
        
        # Penalty for deviation from sphere
        deviations = np.sum(p * p, axis=1, keepdims=True) - 1.0
        penalty = 0.5 * np.sum(deviations * deviations)
        grad_penalty = 2.0 * deviations * p
        
        total_energy = energy + penalty
        total_grad = (grad_manifold + grad_penalty).flatten()
        
        return total_energy, total_grad

    res = minimize(
        lambda x: obj_grad(x)[0],
        pts.flatten(),
        jac=lambda x: obj_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'disp': False}
    )
    
    final_pts = _project_to_sphere(res.x.reshape(N, 3))
    energy, _ = _energy_and_grad(final_pts)
    
    return final_pts, energy

def _cg_sphere(pts, time_limit, maxiter=1000, tol=1e-10):
    """
    Riemannian conjugate gradient on sphere using exponential map.
    """
    start_time = time.time()
    n = N
    
    energy, grad = _energy_and_grad(pts)
    
    # Initial search direction: negative gradient projected to tangent space
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts
    
    # Line search parameters
    alpha = 0.1
    beta = 0.5
    
    for it in range(maxiter):
        if time.time() - start_time > time_limit:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Normalized search direction
        d = -grad_tangent / grad_norm
        
        # Backtracking line search
        best_energy = energy
        best_pts = pts
        for ls in range(20):
            new_pts = _exp_map(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                break
            alpha *= 0.5
        
        if best_energy >= energy:
            # No improvement, try smaller step
            alpha *= 0.1
            if alpha < 1e-12:
                break
            continue
        
        # Update conjugate gradient
        old_grad_norm = grad_norm
        new_dot = np.sum(best_grad * grad, axis=1, keepdims=True)
        new_grad_tangent = best_grad - new_dot * best_pts
        new_grad_norm = np.linalg.norm(new_grad_tangent)
        
        # Polak-Ribiere beta
        beta_coeff = np.sum(new_grad_tangent * (new_grad_tangent - grad_tangent)) / (old_grad_norm * old_grad_norm + 1e-15)
        beta_coeff = max(0.0, beta_coeff)
        
        # Update search direction
        grad_tangent = new_grad_tangent
        d = -grad_tangent + beta_coeff * d * (old_grad_norm / (new_grad_norm + 1e-15))
        
        pts = best_pts
        energy = best_energy
        grad = best_grad
        
        # Reset alpha for next iteration
        alpha = 0.1
    
    return pts, energy

def _alternating_opt(pts, time_limit):
    """
    Alternating optimization: L-BFGS then CG, repeat.
    """
    start_time = time.time()
    best_pts = pts
    best_energy, _ = _energy_and_grad(pts)
    
    for round_num in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # L-BFGS phase
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2.0:
            break
        lbfgs_time = min(remaining * 0.4, 15.0)
        pts, energy = _lbfgs_sphere(pts, lbfgs_time, maxiter=200, tol=1e-10)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # CG phase
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2.0:
            break
        cg_time = min(remaining * 0.6, 30.0)
        pts, energy = _cg_sphere(pts, cg_time, maxiter=500, tol=1e-10)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Generate initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Halton sequence
    inits.append(_halton_sphere(N))
    
    # 3. Random initializations
    for seed in [42, 123, 456, 789]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand = _project_to_sphere(pts_rand)
        inits.append(pts_rand)
    
    # 4. Perturbed Fibonacci
    for seed, strength in [(7, 0.05), (99, 0.1), (123, 0.15)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert = _project_to_sphere(pts_pert)
        inits.append(pts_pert)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Coarse optimization with multiple inits
    coarse_time_per_init = 8.0
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 40:  # Leave 40s for deep polish
            break
        
        remaining = deadline - time.time()
        time_alloc = min(coarse_time_per_init, remaining - 40)
        if time_alloc < 1.0:
            break
        
        pts, energy = _alternating_opt(init_pts, time_alloc)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Deep polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining > 10:
            best_pts, best_energy = _alternating_opt(
                best_pts, remaining - 2, 
            )
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 2:
        remaining = deadline - time.time()
        if remaining > 5:
            # Use L-BFGS with tight tolerance
            best_pts, best_energy = _lbfgs_sphere(
                best_pts, remaining - 1,
                maxiter=500, tol=1e-14
            )
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    else:
        best_pts = _project_to_sphere(best_pts)
    
    # Verify all points are valid and not too close
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
