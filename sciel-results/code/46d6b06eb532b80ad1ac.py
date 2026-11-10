"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from math import log, fsum
import time

N = 150
DEADLINE_FRACTION = 0.85  # Safety margin

def _spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates (theta, phi) to Cartesian (x, y, z).
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2pi]
    """
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(pts):
    """Convert Cartesian points to spherical coordinates (theta, phi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-15), -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2pi)
    phi = phi % (2 * np.pi)
    return theta, phi

def _compute_energy_and_grad_spherical(theta, phi):
    """
    Compute logarithmic energy and gradient in spherical coordinates.
    E = sum_{i<j} -log(||xi - xj||)
    dE/dtheta_i = sum_{j!=i} (z_i - z_j) / (r_i r_j ||xi - xj||^2) * (x_i - x_j) . (dxi/dtheta) ... 
    Actually, it's easier to compute gradient in Cartesian and project, 
    or derive directly. Let's derive directly for efficiency.
    
    Let x_i = (sin(theta_i)cos(phi_i), sin(theta_i)sin(phi_i), cos(theta_i))
    ||x_i - x_j||^2 = 2 - 2(sin(theta_i)sin(theta_j)cos(phi_i-phi_j) + cos(theta_i)cos(theta_j))
    Let D_ij = ||x_i - x_j||^2.
    Energy term: -log(sqrt(D_ij)) = -0.5 log(D_ij).
    
    Gradient with respect to theta_i:
    dE/dtheta_i = sum_{j!=i} -0.5 * (1/D_ij) * dD_ij/dtheta_i
    
    dD_ij/dtheta_i = d/dtheta_i [2 - 2(sin(theta_i)sin(theta_j)cos(phi_i-phi_j) + cos(theta_i)cos(theta_j))]
    = -2 [cos(theta_i)sin(theta_j)cos(phi_i-phi_j) - sin(theta_i)cos(theta_j)]
    
    Gradient with respect to phi_i:
    dE/dphi_i = sum_{j!=i} -0.5 * (1/D_ij) * dD_ij/dphi_i
    
    dD_ij/dphi_i = d/dphi_i [2 - 2(sin(theta_i)sin(theta_j)cos(phi_i-phi_j) + cos(theta_i)cos(theta_j))]
    = -2 [ -sin(theta_i)sin(theta_j)(-sin(phi_i-phi_j)) ]
    = 2 sin(theta_i)sin(theta_j)sin(phi_i-phi_j)
    
    We need to sum over j != i.
    """
    n = len(theta)
    # Compute pairwise differences
    # theta[i, j] = theta[i] - theta[j]
    d_theta = theta[:, np.newaxis] - theta[np.newaxis, :]
    d_phi = phi[:, np.newaxis] - phi[np.newaxis, :]
    
    # Compute D_ij = ||xi - xj||^2
    # D = 2 - 2(sin(th_i)sin(th_j)cos(d_phi) + cos(th_i)cos(th_j))
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    
    sin_prod = sin_th[:, np.newaxis] * sin_th[np.newaxis, :]
    cos_prod = cos_th[:, np.newaxis] * cos_th[np.newaxis, :]
    cos_d_phi = np.cos(d_phi)
    
    dot_prod = sin_prod * cos_d_phi + cos_prod
    D = 2.0 - 2.0 * dot_prod
    
    # Avoid division by zero for i=j (D=0)
    np.fill_diagonal(D, 1.0) # Placeholder, will be masked
    inv_D = 1.0 / D
    np.fill_diagonal(inv_D, 0.0)
    
    # Energy
    # E = sum_{i<j} -0.5 log(D_ij)
    # We can compute sum over all i!=j and divide by 2
    # log(D) is singular at 0, but we masked diagonal.
    # Off-diagonal D > 0.
    log_D = np.log(D)
    np.fill_diagonal(log_D, 0.0)
    energy = -0.25 * np.sum(log_D) # -0.5 * 0.5 (for i<j) * 2 (for i!=j sum) = -0.25 sum_{i!=j}
    # Wait. Sum_{i<j} f(i,j) = 0.5 Sum_{i!=j} f(i,j).
    # f(i,j) = -0.5 log(D_ij).
    # So E = 0.5 * Sum_{i!=j} (-0.5 log D_ij) = -0.25 Sum_{i!=j} log D_ij.
    # Correct.
    
    # Gradient dE/dtheta_i
    # dD_ij/dtheta_i = -2 [cos(theta_i)sin(theta_j)cos(d_phi) - sin(theta_i)cos(theta_j)]
    # Let's compute the term inside the sum for each i.
    # term_theta_ij = -0.5 * (1/D_ij) * dD_ij/dtheta_i
    # dD_ij/dtheta_i = -2 (cos_th_i sin_th_j cos_d_phi - sin_th_i cos_th_j)
    # So term_theta_ij = (1/D_ij) (cos_th_i sin_th_j cos_d_phi - sin_th_i cos_th_j)
    
    # We need sum_j term_theta_ij for each i.
    # term_theta_ij = (cos_th_i sin_th_j cos_d_phi - sin_th_i cos_th_j) / D_ij
    # This is not separable easily due to D_ij.
    # Let's compute the matrix of terms.
    
    # cos_th_i is row i, sin_th_j is col j.
    # cos_d_phi is matrix.
    # Term 1: cos_th_i * sin_th_j * cos_d_phi / D_ij
    # Term 2: - sin_th_i * cos_th_j / D_ij
    
    # Let's vectorize.
    # A_ij = cos_th_i sin_th_j cos_d_phi
    A = cos_th[:, np.newaxis] * sin_th[np.newaxis, :] * cos_d_phi
    # B_ij = sin_th_i cos_th_j
    B = sin_th[:, np.newaxis] * cos_th[np.newaxis, :]
    
    # grad_theta_i = sum_j (A_ij - B_ij) * inv_D_ij
    # Note: inv_D has 0 on diagonal.
    grad_theta = np.sum((A - B) * inv_D, axis=1)
    
    # Gradient dE/dphi_i
    # dD_ij/dphi_i = 2 sin(theta_i)sin(theta_j)sin(d_phi)
    # term_phi_ij = -0.5 * (1/D_ij) * 2 sin(theta_i)sin(theta_j)sin(d_phi)
    #              = - sin(theta_i)sin(theta_j)sin(d_phi) / D_ij
    # grad_phi_i = sum_j term_phi_ij
    
    C = sin_th[:, np.newaxis] * sin_th[np.newaxis, :] * np.sin(d_phi)
    grad_phi = -np.sum(C * inv_D, axis=1)
    
    return energy, grad_theta, grad_phi

def _gradient_flow(theta, phi, time_budget, rng):
    """
    Perform gradient flow in spherical coordinates.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Adaptive step size
    # Initial step size
    step_size = 0.01
    # Lipschitz constant estimate (rough)
    # The gradient can be large if points are close.
    # We use a backtracking line search to ensure descent.
    
    current_energy, grad_th, grad_ph = _compute_energy_and_grad_spherical(theta, phi)
    best_theta = theta.copy()
    best_phi = phi.copy()
    best_energy = current_energy
    
    # Simulated annealing style perturbation in early phase
    # Phase 1: High temp, larger steps, more restarts/perturbations
    # Phase 2: Low temp, small steps, precise convergence
    
    # We will run a loop that checks time.
    # To allow for "multi-restart" within the function, we can just run the flow.
    # The outer loop in construct() handles multiple starts.
    
    # Let's add a simple perturbation mechanism if stuck.
    # But for now, pure gradient flow with adaptive step.
    
    # Initial step size based on gradient norm
    grad_norm = np.sqrt(np.sum(grad_th**2) + np.sum(grad_ph**2))
    if grad_norm > 1e-10:
        step_size = 0.1 / grad_norm
    else:
        step_size = 0.01
        
    step_size = np.clip(step_size, 1e-4, 0.1)
    
    for _ in range(10000): # Max iterations safety
        if time.time() > deadline:
            break
            
        # Compute gradient
        current_energy, grad_th, grad_ph = _compute_energy_and_grad_spherical(theta, phi)
        
        # Update
        new_theta = theta - step_size * grad_th
        new_phi = phi - step_size * grad_ph
        
        # Wrap phi
        new_phi = new_phi % (2 * np.pi)
        # Clip theta to avoid poles (numerical stability)
        new_theta = np.clip(new_theta, 1e-6, np.pi - 1e-6)
        
        # Check energy
        new_energy, _, _ = _compute_energy_and_grad_spherical(new_theta, new_phi)
        
        # Backtracking line search
        if new_energy < current_energy:
            theta = new_theta
            phi = new_phi
            current_energy = new_energy
            # Increase step size slightly if we had a big drop
            # step_size = np.clip(step_size * 1.05, 1e-4, 0.1)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_theta = theta.copy()
                best_phi = phi.copy()
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                # Converged or stuck
                break
                
    return best_theta, best_phi, best_energy

def _repulsive_init_spherical(n, rng):
    """
    Initialize points using repulsive forces in spherical space.
    """
    # Start with random points
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    
    # Run a few steps of gradient ascent on energy (minimize -energy)
    # Or just use gradient flow to minimize energy, which naturally repels.
    # Since we have a time budget, we can do a quick optimization here.
    
    # Let's just return the random init for now, and let the main flow do the work.
    # A better init: Fibonacci spiral converted to spherical.
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # Convert to spherical
    th = np.arccos(np.clip(z, -1.0, 1.0))
    ph = np.arctan2(y, x) % (2 * np.pi)
    
    return th, ph

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    best_theta = None
    best_phi = None
    best_energy = 1e30
    
    # Multi-restart strategy
    # 1. Fibonacci spiral
    # 2. Random
    # 3. Random with jitter
    
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            # Fibonacci
            init_theta, init_phi = _repulsive_init_spherical(N, rng)
        else:
            # Random
            init_theta = rng.uniform(0.1, np.pi - 0.1, N)
            init_phi = rng.uniform(0, 2 * np.pi, N)
            
            # Add some structure to avoid pure chaos? 
            # Maybe just random is fine with enough restarts.
            
        # Allocate remaining time to this restart
        # We want to spend more time on later restarts if they look promising?
        # For simplicity, split time evenly, but ensure at least 10s per restart.
        time_left = deadline - time.time()
        if time_left <= 0:
            break
            
        # If this is the last restart, use all remaining time.
        # Otherwise, use a fraction.
        if r < n_restarts - 1:
            time_budget = time_left / (n_restarts - r) * 0.9
        else:
            time_budget = time_left * 0.95
            
        # Ensure minimum time
        time_budget = max(time_budget, 1.0)
        
        res_theta, res_phi, res_energy = _gradient_flow(init_theta, init_phi, time_budget, rng)
        
        if res_energy < best_energy:
            best_energy = res_energy
            best_theta = res_theta
            best_phi = res_phi
            
    # Final polish
    if time.time() < deadline - 5 and best_theta is not None:
        # Run gradient flow again with very small step size to polish
        # We can just call _gradient_flow again, it will handle step size.
        # To ensure polish, we can set a smaller initial step size.
        # But _gradient_flow adapts.
        # Let's just run it for the remaining time.
        time_left = deadline - time.time()
        res_theta, res_phi, res_energy = _gradient_flow(best_theta, best_phi, time_left, rng)
        
        if res_energy < best_energy:
            best_energy = res_energy
            best_theta = res_theta
            best_phi = res_phi
            
    # Convert to Cartesian
    if best_theta is None:
        # Fallback
        best_theta, best_phi = _repulsive_init_spherical(N, rng)
        
    pts = _spherical_to_cartesian(best_theta, best_phi)
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts / norms
    
    return pts
