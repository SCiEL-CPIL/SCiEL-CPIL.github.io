"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on sphere."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)  # Avoid log(0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    energy = -np.log(dist).sum() / 2.0  # Each pair counted twice
    
    # Gradient: d/dx_i [log(1/d_ij)] = (x_i - x_j) / d_ij^2
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j (x_i - x_j) / d_ij^2
    sum_inv_dist2 = inv_dist2.sum(axis=1)  # Shape (n,)
    weighted_sum = (inv_dist2[:, :, np.newaxis] * pts[np.newaxis, :, :]).sum(axis=1)
    grad = pts * sum_inv_dist2[:, np.newaxis] - weighted_sum
    
    # Project gradient onto tangent space: remove radial component
    radial_comp = (grad * pts).sum(axis=1, keepdims=True)
    grad_tangent = grad - radial_comp * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize(pts, max_iter=500, tol=1e-10, deadline=None):
    """Optimize points on sphere using L-BFGS-B with manifold constraints."""
    pts = pts.copy()
    
    def objective_and_grad(flat_pts):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        pts_reshaped = flat_pts.reshape(-1, 3)
        pts_reshaped = _project_to_sphere(pts_reshaped)
        energy, grad = _energy_and_grad(pts_reshaped)
        return energy, grad.flatten()
    
    # Use L-BFGS-B optimizer
    try:
        result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B', 
                          jac=True, options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50})
        opt_pts = _project_to_sphere(result.x.reshape(-1, 3))
        opt_energy, _ = _energy_and_grad(opt_pts)
    except (TimeoutError, Exception):
        opt_pts = pts
        opt_energy, _ = _energy_and_grad(opt_pts)
    
    # Final high-precision polish pass with tighter tolerance
    if deadline is None or time.time() < deadline - 3:
        try:
            result2 = minimize(objective_and_grad, opt_pts.flatten(), method='L-BFGS-B',
                               jac=True, options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            polish_pts = _project_to_sphere(result2.x.reshape(-1, 3))
            polish_energy, _ = _energy_and_grad(polish_pts)
            if polish_energy < opt_energy:
                opt_pts = polish_pts
                opt_energy = polish_energy
        except (TimeoutError, Exception):
            pass
    
    return opt_pts, opt_energy

def _simulated_annealing(pts, t_start=0.1, t_end=0.001, max_iter=200, cooling_rate=0.98, deadline=None):
    """Refine with simulated annealing to escape local minima."""
    pts = pts.copy()
    n = len(pts)
    energy, _ = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    t = t_start
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random perturbation in tangent space with adaptive scale
        noise = np.random.randn(n, 3)
        # Project to tangent space
        radial_comp = (noise * pts).sum(axis=1, keepdims=True)
        noise_tangent = noise - radial_comp * pts
        # Scale by temperature with adaptive step size
        step_scale = 0.03 * (t / t_start) ** 0.5
        noise_tangent *= step_scale
        
        new_pts = pts + noise_tangent
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - energy
        
        # Accept with probability exp(-delta_e / t), clamp for numerical stability
        if delta_e < 0 or (t > 1e-6 and np.random.rand() < math.exp(min(0, -delta_e / t))):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        t *= cooling_rate
        if t < t_end:
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
    
    # Rotation matrix
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

def construct():
    start_time = time.time()
    deadline = start_time + 143  # Leave 7s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Multiple restarts with different initializations
    for restart in range(25):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Fibonacci sphere
            pts = _fibonacci_sphere(N, seed=restart)
        elif restart < 4:
            # Random points
            rng = np.random.RandomState(restart * 100 + 42)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart < 8:
            # Fibonacci with different seeds
            pts = _fibonacci_sphere(N, seed=restart + 100)
        elif restart < 12:
            # Random with varying concentration
            rng = np.random.RandomState(restart * 200 + 13)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Clustered starts with varying cluster count
            n_clusters = 3 + (restart % 5)
            rng = np.random.RandomState(restart * 300 + 7)
            centers = rng.randn(n_clusters, 3)
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            pts = []
            for i in range(N):
                center = centers[i % n_clusters]
                noise = rng.randn(3) * 0.15
                pt = center + noise
                pt = pt / np.linalg.norm(pt)
                pts.append(pt)
            pts = np.array(pts)
        
        # Optimize using L-BFGS-B
        try:
            opt_pts, opt_energy = _optimize(pts, max_iter=350, deadline=deadline)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            continue
        
        # Simulated annealing refinement followed by re-optimization
        if time.time() < deadline - 12:
            try:
                sa_pts, sa_energy = _simulated_annealing(opt_pts, t_start=0.06, t_end=0.0003, max_iter=120, deadline=deadline)
                if sa_energy < best_energy:
                    best_energy = sa_energy
                    best_pts = sa_pts.copy()
                
                # Re-optimize from SA result to polish
                if time.time() < deadline - 6:
                    re_opt_pts, re_opt_energy = _optimize(sa_pts, max_iter=200, deadline=deadline)
                    if re_opt_energy < best_energy:
                        best_energy = re_opt_energy
                        best_pts = re_opt_pts.copy()
            except Exception:
                continue
    
    # Final polish on the best result
    if best_pts is not None and time.time() < deadline - 3:
        try:
            final_pts, final_energy = _optimize(best_pts, max_iter=400, tol=1e-14, deadline=deadline)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N, seed=0)
    
    return best_pts
# EVOLVE-BLOCK-END
