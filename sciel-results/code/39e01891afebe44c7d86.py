"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _log_energy_and_grad(points):
    """
    Compute energy and Riemannian gradient using Gram matrix for efficiency.
    Energy: E = -sum_{i<j} log(|x_i - x_j|)
    |x_i - x_j|^2 = 2 - 2 * (x_i . x_j)
    Let G be the Gram matrix. dist_sq_ij = 2 - 2*G_ij.
    Grad_i (Euclidean) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    Note: (x_i - x_j) / |x_i - x_j|^2 = (x_i - x_j) / (2 - 2 G_ij)
    We can compute this efficiently.
    """
    n = points.shape[0]
    # Gram matrix
    G = points @ points.T
    # Distance squared
    dist_sq = 2.0 - 2.0 * G
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero, though we mask it
    
    # Energy
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dist_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient
    # grad_i = sum_{j!=i} (x_i - x_j) / dist_sq_ij
    # = x_i * sum_{j!=i} 1/dist_sq_ij - sum_{j!=i} x_j / dist_sq_ij
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # sum_j 1/dist_sq_ij for each i
    row_sums = np.sum(inv_dist_sq, axis=1)
    
    # term1 = x_i * row_sums
    term1 = points * row_sums[:, np.newaxis]
    
    # term2 = sum_j x_j / dist_sq_ij
    # This is (inv_dist_sq @ points)
    term2 = inv_dist_sq @ points
    
    grad_euclidean = term1 - term2
    
    # Project to tangent space
    dot_products = np.sum(grad_euclidean * points, axis=1, keepdims=True)
    grad_riemannian = grad_euclidean - dot_products * points
    
    return energy, grad_riemannian

def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _md_relaxation(points, max_steps=2000, dt=0.01, gamma=0.1, rng=None):
    """
    Molecular Dynamics relaxation on the sphere.
    Uses forces = -gradient.
    Updates: v = (1-gamma)v + F*dt; x = x + v*dt (projected)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    current_pts = points.copy()
    n = current_pts.shape[0]
    v = np.zeros_like(current_pts)
    
    # Adaptive dt
    current_dt = dt
    
    for step in range(max_steps):
        energy, grad = _log_energy_and_grad(current_pts)
        force = -grad
        
        # Update velocity
        v = (1.0 - gamma) * v + force * current_dt
        
        # Project velocity to tangent space
        dot_v_x = np.sum(v * current_pts, axis=1, keepdims=True)
        v_tangent = v - dot_v_x * current_pts
        
        # Propagate
        new_pts = current_pts + v_tangent * current_dt
        new_pts = _normalize(new_pts)
        
        # Check for energy increase or instability
        new_energy, _ = _log_energy_and_grad(new_pts)
        
        if new_energy > energy + 1e-6:
            # Reject step, reduce dt
            current_dt *= 0.5
            v = 0.5 * v
            if current_dt < 1e-8:
                break
            continue
            
        current_pts = new_pts
        # Slowly increase dt if stable
        if step % 10 == 0:
            current_dt = min(dt * 2.0, current_dt * 1.1)
            
    return current_pts, energy

def _local_refine(points, max_iter=50):
    """
    Fast local refinement using Newton-like steps or strong gradient descent.
    """
    current_pts = points.copy()
    current_energy, current_grad = _log_energy_and_grad(current_pts)
    
    # Simple backtracking line search with Riemannian steps
    step_size = 0.1
    for i in range(max_iter):
        # Compute Hessian approx? Too slow. Use gradient norm to adapt step size.
        grad_norm = np.linalg.norm(current_grad)
        if grad_norm < 1e-8:
            break
            
        # Try step
        v_tangent = -current_grad
        # Normalize step direction to unit length for consistent step size
        v_norm = np.linalg.norm(v_tangent, axis=1, keepdims=True)
        v_norm[v_norm < 1e-12] = 1.0
        dir_vec = v_tangent / v_norm
        
        new_pts = current_pts + dir_vec * step_size
        new_pts = _normalize(new_pts)
        
        new_energy, new_grad = _log_energy_and_grad(new_pts)
        
        if new_energy < current_energy:
            current_pts = new_pts
            current_energy = new_energy
            current_grad = new_grad
            # Increase step size slightly if improvement is good
            step_size = min(0.5, step_size * 1.05)
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
                
    return current_pts, current_energy

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)
    
    rng = np.random.default_rng(123)

    # 1. Initial MD relaxation from Fibonacci
    try:
        md_pts, e = _md_relaxation(best_points, max_steps=1000, dt=0.05, gamma=0.05, rng=rng)
        if e < best_energy:
            best_energy = e
            best_points = md_pts
    except Exception:
        pass

    # 2. Local refinement
    if time.time() < deadline - 20:
        try:
            ref_pts, e = _local_refine(best_points, max_iter=100)
            if e < best_energy:
                best_energy = e
                best_points = ref_pts
        except Exception:
            pass

    # 3. Basin Hopping via MD restarts
    #    Perturb the best solution, run short MD, refine.
    current_best = best_points.copy()
    
    num_restarts = 0
    while time.time() < deadline - 30:
        num_restarts += 1
        # Perturb
        noise = rng.standard_normal((N, 3))
        dot_n_x = np.sum(noise * current_best, axis=1, keepdims=True)
        noise_tangent = noise - dot_n_x * current_best
        noise_tangent *= 0.5 # Moderate perturbation
        
        perturbed = _normalize(current_best + noise_tangent)
        
        # Short MD
        try:
            md_pts, e = _md_relaxation(perturbed, max_steps=500, dt=0.02, gamma=0.1, rng=rng)
            if e < best_energy:
                best_energy = e
                best_points = md_pts
                current_best = md_pts.copy()
                # Reset perturbation scale? Keep it moderate.
            else:
                # If not better, maybe try a different perturbation strength next time
                pass
        except Exception:
            pass

        # If we've found a very good solution, we might want to polish it more
        if time.time() < deadline - 10 and num_restarts % 5 == 0:
             try:
                 ref_pts, e = _local_refine(best_points, max_iter=50)
                 if e < best_energy:
                     best_energy = e
                     best_points = ref_pts
                     current_best = ref_pts.copy()
             except Exception:
                 pass

    # 4. Final Polish
    if time.time() < deadline - 5:
        try:
            final_pts, e = _local_refine(best_points, max_iter=200)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
