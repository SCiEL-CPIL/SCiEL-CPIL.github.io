"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 150

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the sphere (chordal distances).
    
    Energy: sum_{i<j} log(1 / |x_i - x_j|)
    Gradient w.r.t. each point on the sphere (tangent component).
    """
    # pts: (N, 3)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    # Avoid self-interaction and division by zero
    np.fill_diagonal(dist, np.inf)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    log_terms = -np.log(dist[iu])
    energy = np.sum(log_terms)
    
    # Gradient: d/dx_i of log(1/|x_i - x_j|) = -d/dx_i log(|x_i - x_j|)
    # = -(x_i - x_j) / |x_i - x_j|^2 * (-1) ... wait let me redo
    # f = log(1/d) = -log(d), d = |x_i - x_j|
    # df/dx_i = -1/d * d/dx_i (|x_i - x_j|) = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
    # So gradient contribution to x_i from pair (i,j) is -(x_i - x_j)/|x_i - x_j|^2
    # And by symmetry, contribution to x_j is -(x_j - x_i)/|x_i - x_j|^2 = (x_i - x_j)/|x_i - x_j|^2
    
    # For each point i, sum over j != i: -(x_i - x_j) / |x_i - x_j|^2
    # = -x_i * sum_j (1/d_ij^2) + sum_j x_j / d_ij^2
    
    # inv_dist_sq: (N, N), diagonal is 0
    inv_dist_sq = np.zeros_like(dist)
    valid = dist > 1e-15
    inv_dist_sq[valid] = 1.0 / (dist[valid] ** 2)
    
    # grad[i] = sum_j (x_j - x_i) / dist_ij^2 = sum_j x_j * inv_dist_sq[i,j] - x_i * sum_j inv_dist_sq[i,j]
    # Using matrix operations
    # S[i] = sum_j x_j * inv_dist_sq[i,j]
    S = inv_dist_sq @ pts  # (N, 3)
    row_sums = inv_dist_sq.sum(axis=1)  # (N,)
    grad = S - pts * row_sums[:, np.newaxis]
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1)  # (N,)
    grad_tangent = grad - dot[:, np.newaxis] * pts
    
    return energy, grad_tangent


def _project_to_sphere(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    time_limit = 115.0  # safety margin before 120s kill
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_pts = _project_to_sphere(best_pts)
    
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Multi-restart local optimization using L-BFGS-B with line search
    from scipy.optimize import minimize
    
    rng = np.random.default_rng(42)
    
    def objective_and_grad(params):
        """Unconstrained objective: energy computed from projected points."""
        pts = _project_to_sphere(params.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # Chain rule through projection is complex; use numerical gradient or
        # instead optimize on the sphere directly with a different approach.
        # For simplicity, use the tangent gradient as if it's the gradient in R^3
        # (this is a common approximation that works well in practice)
        return e, g.ravel()
    
    # Use multiple random restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        if restart == 0:
            x0 = best_pts.copy()
        else:
            # Random perturbation
            noise = rng.standard_normal((N, 3)) * 0.1
            x0 = _project_to_sphere(best_pts + noise)
        
        # Compute initial energy
        e0, g0 = _energy_and_grad(x0)
        
        try:
            result = minimize(
                objective_and_grad,
                x0.ravel(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_energy, _ = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
                
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    if time.time() - start_time < time_limit - 5.0:
        try:
            def objective_and_grad_patched(params):
                pts = _project_to_sphere(params.reshape(N, 3))
                e, g = _energy_and_grad(pts)
                return e, g.ravel()
            
            result = minimize(
                objective_and_grad_patched,
                best_pts.ravel(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge point j slightly
                best_pts[j] = _project_to_sphere(best_pts[j] + rng.standard_normal(3) * 1e-6)
    
    # Final normalization
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
