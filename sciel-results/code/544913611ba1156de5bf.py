"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = points.shape[0]
    # Compute pairwise distances
    # points: (n, 3)
    # dist2[i,j] = |x_i - x_j|^2
    dist2 = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * points @ points.T
    np.fill_diagonal(dist2, 1.0)  # avoid log(0) on diagonal
    dist = np.sqrt(dist2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # log(1/d) = -log(d)
    return -np.sum(np.log(d))


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy with respect to points.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = points.shape[0]
    # dist2[i,j] = |x_i - x_j|^2
    dist2 = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * points @ points.T
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # (x_i - x_j) / dist2[i,j]
    # We want to compute for each i: sum_j (x_i - x_j) / dist2[i,j]
    # = x_i * sum_j (1/dist2[i,j]) - sum_j (x_j / dist2[i,j])
    
    # 1/dist2, with diagonal set to 0
    inv_dist2 = np.zeros_like(dist2)
    mask = np.ones_like(dist2, dtype=bool)
    np.fill_diagonal(mask, False)
    inv_dist2[mask] = 1.0 / dist2[mask]
    
    # sum_j 1/dist2[i,j]
    sum_inv = np.sum(inv_dist2, axis=1)  # (n,)
    
    # sum_j x_j / dist2[i,j]
    # This is (inv_dist2 @ points) where inv_dist2[i,j] = 1/dist2[i,j]
    sum_x_inv = inv_dist2 @ points  # (n, 3)
    
    grad = points * sum_inv[:, None] - sum_x_inv
    
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    deadline = time.time() + 110.0  # 10s safety margin before 120s limit
    
    # Start with Fibonacci spiral
    best_points = _fibonacci_spiral()
    best_energy = _log_energy(best_points)
    
    # Try multiple random restarts with gradient-based optimization
    np.random.seed(42)
    
    n_restarts = 50  # max number of restarts we'll try
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial points
        if restart == 0:
            points = best_points.copy()
        else:
            # Random points on sphere
            vecs = np.random.randn(N, 3)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            points = vecs
        
        # Use L-BFGS-B with projection, or use scipy's minimize with SLSQP
        # Better: use gradient descent with projection
        
        # We'll do a few optimization strategies:
        # 1. L-BFGS on the sphere using Riemannian-like approach (project after each step)
        # 2. SLSQP with equality constraints
        
        # Strategy: use scipy minimize with SLSQP for equality constraints
        # But SLSQP can be slow. Let's use a simple projected gradient descent + L-BFGS
        
        # Actually, let's use L-BFGS-B on the 3N coordinates with the constraint that
        # points are on the sphere. We can use the trick of parameterizing with angles,
        # or just use projected gradient.
        
        # Simpler: use scipy.optimize.minimize with method='L-BFGS-B' won't handle
        # equality constraints well. Let's use SLSQP for small problems.
        
        # For N=100, we have 300 variables. SLSQP might work but could be slow.
        # Let's try a hybrid: projected gradient for a few steps, then L-BFGS.
        
        # Actually, let's just use L-BFGS on the tangent space approach:
        # At each point, the gradient projected onto the tangent space is:
        # grad_tangent = grad - (grad . x) * x
        # Then take a step in the tangent direction and project back.
        
        # Let's implement a simple Riemannian gradient descent with backtracking.
        
        current_points = points.copy()
        current_energy = _log_energy(current_points)
        
        # Riemannian gradient descent with adaptive step size
        step_size = 0.01
        max_iters = 2000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            grad = _log_energy_grad(current_points)
            
            # Project gradient onto tangent space
            # grad_tangent = grad - (grad . x) * x
            grad_dot_x = np.sum(grad * current_points, axis=1, keepdims=True)
            grad_tangent = grad - grad_dot_x * current_points
            
            # Gradient norm
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-12:
                break
            
            # Backtracking line search
            alpha = step_size
            for _ in range(20):
                new_points = current_points + alpha * grad_tangent
                new_points = _project_to_sphere(new_points)
                new_energy = _log_energy(new_points)
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            
            current_points = new_points
            current_energy = new_energy
            
            # Adaptive step size
            if new_energy < current_energy:
                step_size = min(step_size * 1.2, 0.1)
            else:
                step_size *= 0.8
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current_points.copy()
    
    # Final polish with high-precision optimization on the best solution
    if time.time() < deadline - 20:
        # Use SLSQP for final polish
        def objective_flat(flat_points):
            pts = flat_points.reshape(N, 3)
            return _log_energy(pts)
        
        def grad_flat(flat_points):
            pts = flat_points.reshape(N, 3)
            return _log_energy_grad(pts).flatten()
        
        # Equality constraints: |x_i|^2 = 1 for each i
        constraints = []
        for i in range(N):
            def make_constraint(i):
                def constraint(flat_points):
                    return flat_points[3*i:3*i+3] @ flat_points[3*i:3*i+3] - 1.0
                def constraint_jac(flat_points):
                    jac = np.zeros(3*N)
                    jac[3*i:3*i+3] = 2.0 * flat_points[3*i:3*i+3]
                    return jac
                return {'type': 'eq', 'fun': constraint, 'jac': constraint_jac}
            constraints.append(make_constraint(i))
        
        try:
            result = minimize(
                objective_flat,
                best_points.flatten(),
                jac=grad_flat,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if result.success and result.fun < best_energy:
                best_points = _project_to_sphere(result.x.reshape(N, 3))
                best_energy = _log_energy(best_points)
        except Exception:
            pass
    
    # Ensure points are on the unit sphere
    best_points = _project_to_sphere(best_points)
    
    # Verify validity
    assert best_points.shape == (N, 3)
    assert np.all(np.isfinite(best_points))
    norms = np.linalg.norm(best_points, axis=1)
    assert np.all(norms > 1e-9)
    
    # Check pairwise distances
    dist2 = np.sum(best_points**2, axis=1)[:, None] + np.sum(best_points**2, axis=1)[None, :] - 2.0 * best_points @ best_points.T
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    assert min_dist > 1e-9, f"Minimum distance too small: {min_dist}"
    
    return best_points
# EVOLVE-BLOCK-END
