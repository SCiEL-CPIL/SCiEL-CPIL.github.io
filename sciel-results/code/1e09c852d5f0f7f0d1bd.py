"""New approach: Nesterov Accelerated Gradient on the manifold with Basin Hopping."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math


def _log_energy_vec(points):
    """Compute logarithmic energy using vectorized operations."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    idx = np.triu_indices(N, 1)
    d_pairs = d[idx]
    d_pairs = np.maximum(d_pairs, 1e-15)
    return -np.sum(np.log(d_pairs))


def _log_energy_grad_vec(points):
    """Compute gradient of logarithmic energy using vectorized operations."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    d2_safe = np.where(d2 == 0, 1.0, d2)
    term = -diff / d2_safe[:, :, np.newaxis]
    idx = np.diag_indices(N)
    term[idx] = 0.0
    grad = np.sum(term, axis=1)
    return grad


def _project_to_sphere(points):
    """Project each point to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _nesterov_minimize(points, max_iter=500, step_init=0.1):
    """
    Minimize energy using Nesterov Accelerated Gradient on the sphere manifold.
    """
    x = points.copy()
    v = np.zeros_like(x)
    step = step_init
    mu = 0.9 # Momentum
    
    for it in range(max_iter):
        # Nesterov lookahead
        x_look = x + mu * v
        
        # Compute gradient at lookahead point
        grad = _log_energy_grad_vec(x_look)
        
        # Project gradient to tangent space
        # g_t = g - (g.x)x
        dot = np.sum(grad * x_look, axis=1, keepdims=True)
        grad_t = grad - dot * x_look
        
        # Update velocity
        v = mu * v - step * grad_t
        
        # Update position
        x_new = x + v
        
        # Project to sphere
        x_new = _project_to_sphere(x_new)
        
        # Adaptive step size: if energy increases, reduce step
        e_old = _log_energy_vec(x)
        e_new = _log_energy_vec(x_new)
        
        if e_new < e_old:
            x = x_new
            # Optionally increase step slightly if stable
            # step = min(step * 1.01, step_init)
        else:
            # Reject and reduce step
            step *= 0.5
            v = 0.0 # Reset momentum on rejection
            
        if step < 1e-8:
            break
            
    return x, _log_energy_vec(x)


def _basin_hopping(start_points, max_hops=100, T_start=0.05, T_end=0.001):
    """
    Basin Hopping global optimization using Nesterov local minimizer.
    """
    best_pts = start_points.copy()
    best_energy = _log_energy_vec(best_pts)
    
    # Local minimize the start
    local_pts, local_e = _nesterov_minimize(best_pts, max_iter=200)
    if local_e < best_energy:
        best_pts = local_pts
        best_energy = local_e
        
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    for hop in range(max_hops):
        T = T_start * (T_end / T_start) ** (hop / max_hops)
        
        # Perturb: Pick 2-5 random points and displace them
        n_perturb = np.random.randint(2, 6)
        indices = np.random.choice(N, n_perturb, replace=False)
        
        new_pts = current_pts.copy()
        for idx in indices:
            disp = np.random.randn(3) * 0.1
            new_pts[idx] += disp
        new_pts = _project_to_sphere(new_pts)
        
        # Local minimize the perturbed state
        local_pts, local_e = _nesterov_minimize(new_pts, max_iter=100)
        
        # Acceptance criterion
        delta_e = local_e - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_pts = local_pts
            current_energy = local_e
            
            if current_energy < best_energy:
                best_pts = current_pts.copy()
                best_energy = current_energy
                
    return best_pts, best_energy


def construct():
    np.random.seed(42)
    start_pts = _initial_fibonacci()
    
    # Run Basin Hopping
    best_pts, best_energy = _basin_hopping(start_pts, max_hops=150, T_start=0.05, T_end=0.001)
    
    # Final local refinement
    final_pts, final_e = _nesterov_minimize(best_pts, max_iter=500)
    if final_e < best_energy:
        best_pts = final_pts
        best_energy = final_e
        
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += 1e-6 * np.random.randn(3)
                best_pts = _project_to_sphere(best_pts)
                
    return best_pts
# EVOLVE-BLOCK-END
