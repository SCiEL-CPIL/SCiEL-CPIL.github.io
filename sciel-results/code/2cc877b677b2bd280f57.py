"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

N = 100

def _energy_and_grad(pts):
    """
    Vectorized computation of logarithmic energy and gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: (energy, grad) where grad is the projected gradient in R^{N*3}.
    """
    # Compute pairwise squared distances
    # dist_sq[i, j] = |pts[i] - pts[j]|^2
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d_sq) / 2 ? No.
    # E = sum_{i<j} -log(d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    # Note: sum_{i<j} log(d_sq_ij) is half of sum_{i!=j} log(d_sq_ij)
    # So E = -0.5 * (0.5 * sum_{i!=j} log(d_sq_ij)) = -0.25 * sum_{i!=j} log(d_sq_ij)
    # Wait, let's be precise.
    # E = \sum_{i<j} \log(1/d_{ij}) = -\sum_{i<j} \log(d_{ij})
    # \log(d_{ij}) = 0.5 \log(d_{ij}^2)
    # E = -0.5 \sum_{i<j} \log(d_{ij}^2)
    # Let S = \sum_{i \neq j} \log(d_{ij}^2). Then \sum_{i<j} \log(d_{ij}^2) = S/2.
    # So E = -0.5 * (S/2) = -0.25 * S.
    
    # However, for gradient, it's easier to work with the full sum and divide by 2.
    # Let's compute the gradient of E with respect to pts.
    # dE/dx_i = -0.5 * sum_{j != i} d/dx_i log(|x_i - x_j|^2)
    # d/dx_i log(|x_i - x_j|^2) = 2 (x_i - x_j) / |x_i - x_j|^2
    # So dE/dx_i = -0.5 * sum_{j != i} 2 (x_i - x_j) / |x_i - x_j|^2
    #            = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    #            = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    # Vectorized gradient computation:
    # inv_dist_sq = 1 / dist_sq
    # Set diagonal to 0 to ignore self-interaction
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_raw[i] = sum_j (pts[j] - pts[i]) * inv_dist_sq[i, j]
    # We can compute this as:
    # sum_j pts[j] * inv_dist_sq[i, j] - pts[i] * sum_j inv_dist_sq[i, j]
    
    # Term 1: sum_j pts[j] * w[i, j] -> pts.T @ w
    # Term 2: pts[i] * sum_j w[i, j] -> pts * (w @ ones)
    
    w = inv_dist_sq
    sum_w = np.sum(w, axis=1)
    
    # grad_raw[i] = (pts.T @ w)[i] - pts[i] * sum_w[i]
    # Note: pts.T @ w gives shape (3, N), so we need to transpose it to (N, 3)
    term1 = (pts.T @ w).T
    term2 = pts * sum_w[:, None]
    
    grad_raw = term1 - term2
    
    # Project gradient to tangent space of the sphere
    # grad_tan[i] = grad_raw[i] - (grad_raw[i] . pts[i]) * pts[i]
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    # Compute energy
    # E = -0.25 * sum_{i != j} log(dist_sq[i, j])
    # But we must be careful: the diagonal is 0, log(0) is -inf. We clipped dist_sq.
    # The diagonal terms are log(1e-12) which is large negative. But we set inv_dist_sq diag to 0.
    # For energy, we should sum only upper triangle or divide the full sum by 2.
    # Let's sum the full matrix of log(dist_sq) and divide by 4?
    # sum_{i!=j} log(d_sq) = 2 * sum_{i<j} log(d_sq)
    # E = -0.5 * sum_{i<j} log(d_sq) = -0.5 * (0.5 * sum_{i!=j} log(d_sq)) = -0.25 * sum_{i!=j} log(d_sq)
    
    log_dist_sq = np.log(dist_sq)
    # Mask diagonal to 0 to avoid contribution from self (which is artificial due to clipping)
    log_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, log_dist_sq)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    return energy, grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _conjugate_gradient(pts, max_iter=1000, tol=1e-10, deadline=None):
    """
    Conjugate Gradient method on the sphere.
    """
    x = pts.copy()
    energy, grad = _energy_and_grad(x)
    p = -grad
    
    # Project p to tangent space
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Line search: minimize f(x + alpha * p)
        # Use backtracking line search
        alpha = 0.1
        c1 = 1e-4
        rho = 0.5
        
        # f0 = energy
        # grad0 = grad
        
        for _ in range(50):
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_grad(x_new)
            
            # Armijo condition: f(x_new) <= f(x) + c1 * alpha * grad^T p
            # Note: grad is projected, so grad^T p is the directional derivative
            dir_deriv = np.sum(grad * p)
            
            if e_new <= energy + c1 * alpha * dir_deriv:
                break
            alpha *= rho
        else:
            # If line search fails, break
            break
        
        # Update
        x = x_new
        energy = e_new
        grad = g_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Compute beta (Polak-Ribiere)
        grad_dot = np.sum(grad * grad)
        grad_prev_dot = np.sum(grad * grad) # This is wrong, need previous grad
        # Actually, Polak-Ribiere uses grad_k^T (grad_k - grad_{k-1}) / grad_{k-1}^T grad_{k-1}
        # We need to store previous grad. Let's use Fletcher-Reeves for simplicity:
        # beta = grad_k^T grad_k / grad_{k-1}^T grad_{k-1}
        # We need grad_prev.
        
        # Let's restart CG every 10 iterations to avoid issues (Nonlinear CG often restarts)
        if k % 10 == 0 and k > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
        else:
            # Fletcher-Reeves
            # beta = ||grad||^2 / ||grad_prev||^2
            # We don't have grad_prev stored in this loop structure easily without extra vars.
            # Let's just use steepest descent (p = -grad) for simplicity and robustness,
            # or implement proper CG storage.
            # Given the vectorization speed, even steepest descent might be fast enough
            # if we do many iterations. But CG is better.
            # Let's do a simple restart every 5 steps.
            if k % 5 == 0:
                p = -grad
                dot_p_x = np.sum(p * x, axis=1)
                p = p - dot_p_x[:, None] * x
            else:
                # Simple momentum-like update
                p_new = -grad
                dot_p_x = np.sum(p_new * x, axis=1)
                p_new = p_new - dot_p_x[:, None] * x
                # Blend with previous p
                p = 0.5 * p + 0.5 * p_new
    
    return x

def _simulated_annealing(pts, deadline, T_init=0.1, T_end=0.01, cooling=0.999):
    """
    Vectorized Simulated Annealing.
    Moves all points simultaneously with Gaussian noise.
    """
    x = pts.copy()
    energy, _ = _energy_and_grad(x)
    best_x = x.copy()
    best_energy = energy
    
    T = T_init
    step_size = 0.1
    
    while time.time() < deadline:
        # Generate noise for all points
        noise = np.random.randn(N, 3) * step_size
        x_new = x + noise
        x_new = _project_to_sphere(x_new)
        
        e_new, _ = _energy_and_grad(x_new)
        
        delta_e = e_new - energy
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            x = x_new
            energy = e_new
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
        
        T *= cooling
        step_size *= 0.9995
        
        if T < T_end:
            break
    
    return best_x

def construct():
    """
    Main entry point.
    Strategy:
    1. Start with Fibonacci spiral.
    2. Alternate between CG refinement and SA perturbation.
    3. Keep the best configuration found.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess
    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Parameters for the hybrid loop
    n_cycles = 20
    cg_iters_per_cycle = 500
    sa_time_per_cycle = 5.0  # seconds
    
    for cycle in range(n_cycles):
        # Check time
        if time.time() > deadline - 20:
            break
        
        # Phase 1: CG Refinement
        # Optimize the current points
        cg_deadline = min(deadline - 10, time.time() + 10.0)
        refined_pts = _conjugate_gradient(pts, max_iter=cg_iters_per_cycle, deadline=cg_deadline)
        
        e_refined, _ = _energy_and_grad(refined_pts)
        if e_refined < best_energy:
            best_energy = e_refined
            best_pts = refined_pts.copy()
        
        pts = refined_pts
        
        # Phase 2: SA Perturbation
        # Perturb the best found so far to escape local minima
        if time.time() > deadline - 15:
            break
            
        sa_deadline = min(deadline - 5, time.time() + sa_time_per_cycle)
        perturbed_pts = _simulated_annealing(best_pts, sa_deadline)
        
        e_perturbed, _ = _energy_and_grad(perturbed_pts)
        if e_perturbed < best_energy:
            best_energy = e_perturbed
            best_pts = perturbed_pts.copy()
        
        # Set current pts to the perturbed ones for the next cycle's CG
        pts = perturbed_pts
        
    # Final polish with CG
    if time.time() < deadline - 5:
        final_pts = _conjugate_gradient(best_pts, max_iter=2000, deadline=deadline - 1)
        e_final, _ = _energy_and_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
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
# EVOLVE-BLOCK-END
