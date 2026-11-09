"""Rewritten program using L-BFGS-B with manifold gradient handling."""
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
    pts: (N, 3) array of points assumed to be on the unit sphere (or normalized inside).
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
    """
    # Ensure points are normalized for calculation
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    u = pts / norms
    
    # Pairwise differences
    diff = u[:, None, :] - u[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Clip to avoid log(0) and division by 0
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy calculation
    # E = sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d_sq) * 0.5? No.
    # E = sum_{i<j} -log(d_ij) = -0.5 sum_{i<j} log(d_ij^2)
    # Let S = sum_{i!=j} log(d_sq_ij). Then sum_{i<j} log(d_sq_ij) = S/2.
    # E = -0.5 * (S/2) = -0.25 * S.
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient calculation
    # dE/du_i = sum_{j!=i} (u_j - u_i) / ||u_i - u_j||^2
    # Note: The gradient of -log(||u_i - u_j||) w.r.t u_i is -(u_i - u_j)/||u_i - u_j||^2 = (u_j - u_i)/||u_i - u_j||^2
    # Since E = sum -log(d), dE = sum -(1/d) dd.
    # d(d_ij) = (u_i - u_j) . du_i / d_ij.
    # dE/d u_i = sum_{j!=i} - (u_i - u_j) / ||u_i - u_j||^2 = sum_{j!=i} (u_j - u_i) / ||u_i - u_j||^2.
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # We want sum_j w_ij (u_j - u_i) where w_ij = 1/||u_i - u_j||^2
    # sum_j w_ij u_j - u_i sum_j w_ij
    # Matrix form: (W^T u)^T - u (W 1)^T
    # W is symmetric, so W^T = W.
    
    w_sum = np.sum(inv_dist_sq, axis=1)
    
    # term1[i] = sum_j inv_dist_sq[i,j] * u[j]
    term1 = inv_dist_sq @ u
    # term2[i] = u[i] * w_sum[i]
    term2 = u * w_sum[:, None]
    
    grad_raw = term1 - term2
    
    # Project to tangent space: grad_tan = grad_raw - (grad_raw . u_i) u_i
    dot = np.sum(grad_raw * u, axis=1)
    grad_tan = grad_raw - dot[:, None] * u
    
    return energy, grad_tan, u

def _objective_and_grad(x_flat):
    """
    Objective function and gradient for scipy.minimize.
    x_flat: (3N,) vector of unnormalized coordinates.
    """
    x = x_flat.reshape(N, 3)
    
    # Get energy and tangent gradient, and the normalized points u
    energy, grad_tan, u = _energy_and_tangent_grad(x)
    
    # Compute gradient in R^3 space for the normalized points
    # f(x) = E(x/||x||)
    # df/dx = (dE/du) . (du/dx)
    # du_i/dx_i = (1/||x_i||) I - (1/||x_i||^3) x_i x_i^T
    # grad_tan is orthogonal to u_i, so (grad_tan . u_i) = 0.
    # Thus, grad_tan . ( - x_i x_i^T / ||x_i||^3 ) = 0.
    # So grad_x_i = grad_tan_i / ||x_i||
    
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    
    grad_x = grad_tan / norms
    
    return energy, grad_x.flatten()

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_lbfgs(start_pts, deadline, is_polish=False):
    """
    Optimize using L-BFGS-B.
    If is_polish is True, use tighter tolerances and more iterations.
    """
    x0 = start_pts.flatten()
    
    if is_polish:
        # Tighter tolerances for the final polish
        options = {
            'maxiter': 5000, 
            'maxfun': 20000, 
            'ftol': 1e-15, 
            'gtol': 1e-14, 
            'iprint': 0
        }
    else:
        # Standard tolerances for the search phase
        options = {
            'maxiter': 2000, 
            'maxfun': 10000, 
            'ftol': 1e-12, 
            'gtol': 1e-10, 
            'iprint': 0
        }
    
    # Check if we have enough time for the polish phase
    # L-BFGS-B is fast, but 5000 iterations might take a few seconds.
    # We assume the caller checks time before invoking polish.
    
    res = minimize(
        _objective_and_grad, 
        x0, 
        jac=True, 
        method='L-BFGS-B', 
        options=options
    )
    
    return res.x.reshape(N, 3)

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Fibonacci spiral
    fib = _initial_fibonacci()
    inits.append(fib)
    
    # 2. Random points
    for _ in range(8):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    # 3. Jittered Fibonacci
    for _ in range(5):
        pts = fib + rng.normal(size=(N, 3)) * 0.2
        inits.append(_project_to_sphere(pts))
        
    # 4. Anti-symmetric random pairs (exploits symmetry of the sphere)
    # Generate 50 random points, then add their negatives
    # This ensures a good spread and potentially lower energy due to symmetry
    pts_half = rng.normal(size=(N//2, 3))
    pts_full = np.vstack([pts_half, -pts_half])
    inits.append(_project_to_sphere(pts_full))
    
    # 5. Another set of anti-symmetric with different seed
    rng2 = np.random.default_rng(123)
    pts_half2 = rng2.normal(size=(N//2, 3))
    pts_full2 = np.vstack([pts_half2, -pts_half2])
    inits.append(_project_to_sphere(pts_full2))
        
    best_pts = inits[0].copy()
    best_energy, _, _ = _energy_and_tangent_grad(best_pts)
    
    # Optimize each initial configuration
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        try:
            optimized_pts = _optimize_lbfgs(init_pts, deadline, is_polish=False)
            optimized_pts = _project_to_sphere(optimized_pts)
            
            e_opt, _, _ = _energy_and_tangent_grad(optimized_pts)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = optimized_pts.copy()
                
        except Exception:
            continue
        
        # If we are getting close to deadline, stop adding new restarts
        if time.time() > deadline - 25:
            break

    # Final high-precision polish on the best found
    # We reserve at least 10 seconds for this phase to allow for convergence with tight tolerances
    if time.time() < deadline - 10:
        try:
            # Run a high-precision polish
            final_pts = _optimize_lbfgs(best_pts, deadline, is_polish=True)
            final_pts = _project_to_sphere(final_pts)
            e_final, _, _ = _energy_and_tangent_grad(final_pts)
            
            # Also try a small jitter + polish to escape potential shallow local minima
            # This is a common technique in geometric optimization
            if time.time() < deadline - 5:
                jittered = best_pts + rng.normal(size=(N, 3)) * 1e-4
                jittered = _project_to_sphere(jittered)
                jittered_opt = _optimize_lbfgs(jittered, deadline, is_polish=True)
                jittered_opt = _project_to_sphere(jittered_opt)
                e_jitter, _, _ = _energy_and_tangent_grad(jittered_opt)
                
                if e_jitter < e_final:
                    e_final = e_jitter
                    final_pts = jittered_opt

            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
        except Exception:
            pass

    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _initial_fibonacci()
    
    return best_pts

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms
# EVOLVE-BLOCK-END
