import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy_fast(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|).
    Returns (energy, gradient) for optimization.
    """
    n = len(pts)
    # Ensure points are on sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf'), None
    terms = -0.5 * np.log(d_sq)
    energy = math.fsum(terms.tolist())
    
    # Gradient calculation for the normalized points (tangent space)
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq_full = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq_full, 1.0)
    weights = 1.0 / dist_sq_full
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - proj_factor * pts
    
    return energy, grad_tangent

def _objective_and_grad(y_flat):
    """
    Objective function for L-BFGS-B.
    Input: y_flat (3N,)
    Output: energy, gradient (3N,)
    """
    n = len(y_flat) // 3
    y = y_flat.reshape(n, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    pts = y / norms
    
    # Compute energy and tangent gradient
    energy, grad_tangent = _log_energy_fast(pts)
    
    if energy == float('inf'):
        return energy, np.zeros_like(y_flat)
        
    # Map tangent gradient back to raw space gradient
    # dE/dy_i = grad_tangent_i / ||y_i||
    grad_raw = grad_tangent / norms
    
    return energy, grad_raw.flatten()

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, grad, step_size):
    """
    Update points on the sphere using the exponential map.
    This ensures the result stays exactly on the sphere (up to floating point precision)
    and handles the curvature correctly, unlike simple Euclidean projection.
    """
    # grad is in tangent space: grad_i dot pts_i = 0
    # Update rule: x_new = cos(|v|) x + sin(|v|)/|v| v
    # where v = -step_size * grad
    
    v = -step_size * grad
    v_norms = np.linalg.norm(v, axis=1, keepdims=True)
    
    # Avoid division by zero for zero gradients
    mask = v_norms > 1e-12
    
    # For small v, cos(v_norm) ~ 1, sin(v_norm)/v_norm ~ 1
    # We can use Taylor expansion for stability if needed, but np functions are usually fine
    # However, we need to handle the 0/0 case in sin(v)/v
    
    pts_new = np.zeros_like(pts)
    
    # Case 1: v is effectively zero
    pts_new[~mask.flatten()] = pts[~mask.flatten()]
    
    # Case 2: v is non-zero
    if np.any(mask.flatten()):
        v_m = v[mask.flatten()]
        v_norms_m = v_norms[mask.flatten()]
        pts_m = pts[mask.flatten()]
        
        cos_term = np.cos(v_norms_m)
        sin_term = np.sin(v_norms_m) / v_norms_m
        
        pts_new_m = cos_term * pts_m + sin_term * v_m
        
        pts_new[mask.flatten()] = pts_new_m
        
    # Normalize to correct for any floating point drift
    return _project_to_sphere(pts_new)

def construct():
    global DEADLINE
    start_time = time.time()
    
    # 1. Generate diverse initial candidates
    # We use multiple strategies to cover different basins
    candidates = []
    
    # Fibonacci (structured)
    candidates.append(_fibonacci_sphere(N))
    
    # Random seeds
    for seed in [1, 2, 3, 4, 5]:
        candidates.append(_random_uniform_sphere(N, seed))
        
    # "Clustered" random: pick a center, distribute around it
    # This might help escape symmetry traps if any exist
    # But for log energy, uniform is usually best. 
    # Let's add a perturbed Fibonacci
    fib = _fibonacci_sphere(N)
    rng = np.random.default_rng(123)
    perturbed_fib = fib + rng.normal(0, 0.1, size=(N, 3))
    candidates.append(_project_to_sphere(perturbed_fib))

    best_energy = float('inf')
    best_pts = None
    
    # 2. Phase 1: Batched Gradient Descent with Exponential Map
    # We run GD for a fixed number of iterations on all candidates in parallel (vectorized where possible, 
    # but since each candidate is independent, we loop over candidates but vectorize over points).
    
    # Parameters for GD
    max_gd_iters = 200
    initial_lr = 0.01
    decay_rate = 0.99
    
    # We will process candidates one by one to manage memory and time, 
    # but we can vectorize the energy calculation within each candidate.
    
    for idx, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:  # Leave time for L-BFGS
            break
            
        pts = init_pts.copy()
        current_energy, grad = _log_energy_fast(pts)
        
        if not np.isfinite(current_energy):
            continue
            
        lr = initial_lr
        
        for it in range(max_gd_iters):
            if time.time() - start_time > DEADLINE - 20:
                break
                
            # Exponential map update
            pts = _exponential_map(pts, grad, lr)
            
            # Recompute energy and gradient
            new_energy, new_grad = _log_energy_fast(pts)
            
            if not np.isfinite(new_energy):
                # If it breaks, revert and break
                break
                
            if new_energy < current_energy:
                current_energy = new_energy
                grad = new_grad
            else:
                # If energy increases, we might have stepped too far.
                # For simplicity in this batched approach, we just decay LR and retry step.
                # But exponential map should be stable for small steps.
                # Let's just decay LR.
                lr *= 0.9
                # Revert pts to before step? We already updated pts.
                # This is a simple heuristic. A proper line search would be better.
                # Given the time constraint, we accept this simple approach.
                # If it gets stuck, L-BFGS will fix it.
                pass
            
            # Decay learning rate
            lr *= decay_rate
            
            # Early stop if gradient is small
            if np.linalg.norm(grad) < 1e-8:
                break
                
        # Update global best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()
            
        # Keep the current best of this candidate for L-BFGS phase
        # We store all "good" candidates for L-BFGS
        # To save memory/time, we only keep the top 3 candidates after GD
        # We'll store them in a list
        # Let's use a simple list of (energy, pts)
        # We need to define this list outside the loop
        
    # Wait, the above loop structure doesn't easily keep top K. 
    # Let's refactor to keep a list of results.
    
    # Refactored Plan:
    # 1. Run GD on all candidates.
    # 2. Collect results.
    # 3. Sort by energy.
    # 4. Run L-BFGS on top 3.
    
    # Let's re-implement the loop properly.
    
    results = []
    
    for idx, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
            
        pts = init_pts.copy()
        current_energy, grad = _log_energy_fast(pts)
        
        if not np.isfinite(current_energy):
            continue
            
        lr = initial_lr
        
        for it in range(max_gd_iters):
            if time.time() - start_time > DEADLINE - 20:
                break
                
            pts = _exponential_map(pts, grad, lr)
            
            new_energy, new_grad = _log_energy_fast(pts)
            
            if not np.isfinite(new_energy):
                break
                
            current_energy = new_energy
            grad = new_grad
            lr *= decay_rate
            
            if np.linalg.norm(grad) < 1e-8:
                break
                
        results.append((current_energy, pts))
        
    # Sort results
    results.sort(key=lambda x: x[0])
    
    # Update global best from GD phase
    if results:
        if results[0][0] < best_energy:
            best_energy = results[0][0]
            best_pts = results[0][1].copy()
            
    # 3. Phase 2: L-BFGS Polish on top candidates
    # We pick the top 3 unique candidates (if available)
    
    top_k = 3
    # Ensure we don't have duplicates (unlikely with different inits, but check just in case)
    unique_candidates = []
    seen = set()
    for energy, pts in results:
        # Simple check: hash of rounded points
        key = tuple(np.round(pts, 6).flatten())
        if key not in seen:
            seen.add(key)
            unique_candidates.append((energy, pts))
            if len(unique_candidates) >= top_k:
                break
            
    for energy, pts in unique_candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        try:
            init_y = pts.flatten()
            res = minimize(
                _objective_and_grad,
                init_y,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'maxfun': 100,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            y_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = y_opt / norms
            
            e, _ = _log_energy_fast(pts_opt)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = pts_opt.copy()
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization and check
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
