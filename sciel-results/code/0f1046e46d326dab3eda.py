"""Logarithmic energy minimization of 150 points on the unit sphere."""
import numpy as np
import time
import math

N = 150
DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _log_energy_and_grad(pts):
    """Compute log(1/|x_i - x_j|) energy sum over i<j and gradient.
    
    pts: (N, 3) array on unit sphere.
    Returns: (energy, grad) where grad is (N, 3).
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    # Mask diagonal and use upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]  # (n*(n-1)/2,)
    energy = np.sum(np.log(1.0 / d))
    
    # Gradient: d/dx_i sum_{j != i} log(1/|x_i - x_j|) = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: log(1/d) = -log(d), dE/dx_i = -sum_j (1/d_ij) * (x_i - x_j)/d_ij = -sum_j (x_i-x_j)/d_ij^2
    # Wait: d/dx_i log(1/|x_i-x_j|) = d/dx_i (-log|x_i-x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    # So gradient of total energy w.r.t. x_i is -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # For minimization, we want to move in direction of -grad = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    
    # Compute gradient efficiently
    # grad[i] = -sum_{j != i} (pts[i] - pts[j]) / dist[i,j]^2
    # = sum_{j != i} (pts[j] - pts[i]) / dist[i,j]^2
    
    inv_d2 = np.zeros_like(dist)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[mask] = 1.0 / (d ** 2)
    
    # grad[i] = sum_j inv_d2[i,j] * (pts[j] - pts[i])
    # = sum_j inv_d2[i,j] * pts[j] - pts[i] * sum_j inv_d2[i,j]
    row_sum = inv_d2.sum(axis=1)  # (n,)
    grad = inv_d2 @ pts - pts * row_sum[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _velocity_verlet_step(pts, vel, dt, gamma):
    """One velocity-Verlet step with damping for rigid body on sphere.
    
    Uses the fact that on a sphere, the acceleration has both a tangential
    component (from the force) and a normal component (centripetal, which
    is handled by projection). We integrate the tangential dynamics.
    """
    n = pts.shape[0]
    
    # Compute forces (negative gradient of energy)
    _, grad = _log_energy_and_grad(pts)
    # Force = -grad (we want to minimize energy)
    force = -grad
    
    # Project force to tangent space: f_tan = f - (f·x)x
    dot_fx = np.sum(force * pts, axis=1, keepdims=True)
    f_tan = force - dot_fx * pts
    
    # Velocity Verlet:
    # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    # v(t+dt/2) = v(t) + 0.5*a(t)*dt
    # Then recompute a at new position, v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
    
    # Half-step velocity
    v_half = vel + 0.5 * f_tan * dt
    
    # Full-step position
    pts_new = pts + v_half * dt
    
    # Project to sphere
    pts_new = _project_to_sphere(pts_new)
    
    # Recompute force at new position
    _, grad_new = _log_energy_and_grad(pts_new)
    force_new = -grad_new
    dot_fx_new = np.sum(force_new * pts_new, axis=1, keepdims=True)
    f_tan_new = force_new - dot_fx_new * pts_new
    
    # Full-step velocity with damping
    vel_new = v_half + 0.5 * f_tan_new * dt
    vel_new *= gamma  # damping
    
    # Project velocity to tangent space
    dot_vx = np.sum(vel_new * pts_new, axis=1, keepdims=True)
    vel_new = vel_new - dot_vx * pts_new
    
    return pts_new, vel_new


def _refine_lbfgs(pts, max_iter=500, tol=1e-12):
    """Use L-BFGS-B style optimization with projection to refine points."""
    from scipy.optimize import minimize
    
    def energy_fn(x):
        pts = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_and_grad(pts)
        return e
    
    def grad_fn(x):
        pts = _project_to_sphere(x.reshape(-1, 3))
        _, g = _log_energy_and_grad(pts)
        return g.flatten()
    
    x0 = pts.flatten()
    result = minimize(energy_fn, x0, jac=grad_fn, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    pts_opt = _project_to_sphere(result.x.reshape(-1, 3))
    return pts_opt


def construct():
    t_start = time.time()
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    vel = np.zeros_like(pts)
    
    # Initial energy
    e_best, _ = _log_energy_and_grad(pts)
    pts_best = pts.copy()
    
    # Adaptive timestep velocity-Verlet with damping
    dt = 0.01
    gamma = 0.999  # damping factor per step
    
    # Phase 1: Velocity-Verlet annealing
    # Start with larger dt, smaller damping, then reduce
    schedule = [
        # (n_steps, dt, gamma)
        (200, 0.05, 0.99),
        (200, 0.03, 0.995),
        (200, 0.02, 0.997),
        (300, 0.01, 0.998),
        (300, 0.005, 0.999),
        (500, 0.002, 0.9995),
        (500, 0.001, 0.9998),
    ]
    
    for n_steps, dt_val, gamma_val in schedule:
        for step in range(n_steps):
            if time.time() - t_start > DEADLINE - 30:
                break
            pts, vel = _velocity_verlet_step(pts, vel, dt_val, gamma_val)
            
            # Track best
            e_curr, _ = _log_energy_and_grad(pts)
            if e_curr < e_best:
                e_best = e_curr
                pts_best = pts.copy()
        
        if time.time() - t_start > DEADLINE - 30:
            break
    
    # Phase 2: L-BFGS refinement
    if time.time() - t_start < DEADLINE - 10:
        pts_refined = _refine_lbfgs(pts_best, max_iter=300, tol=1e-10)
        e_ref, _ = _log_energy_and_grad(pts_refined)
        if e_ref < e_best:
            e_best = e_ref
            pts_best = pts_refined
    
    # Phase 3: Final high-precision polish
    if time.time() - t_start < DEADLINE - 5:
        pts_polish = _refine_lbfgs(pts_best, max_iter=500, tol=1e-14)
        e_pol, _ = _log_energy_and_grad(pts_polish)
        if e_pol < e_best:
            e_best = e_pol
            pts_best = pts_polish
    
    # Ensure valid output
    pts_best = _project_to_sphere(pts_best)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts_best[i] - pts_best[j])
            if d < 1e-9:
                # Nudge point j slightly
                pts_best[j] += 1e-8 * np.random.randn(3)
                pts_best[j] = _project_to_sphere(pts_best[j].reshape(1, 3))[0]
    
    return pts_best
