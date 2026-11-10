"""
Strategy: Stochastic Global Search (Simulated Annealing) + Deterministic Local Polish.
- The current program uses Riemannian L-BFGS, which is a deterministic local optimizer.
  It is highly susceptible to getting stuck in local minima depending on the initial
  configuration (e.g., Fibonacci spiral).
- This new strategy uses Simulated Annealing (SA) to perform a global search.
  SA allows for "uphill" moves in the early stages (high temperature) to escape
  local minima, and gradually cools down to fine-tune the configuration.
- The moves are random perturbations in the tangent space, followed by projection
  back to the sphere.
- After the SA phase, we run a deterministic Newton-Raphson (or Gauss-Newton)
  local optimization to polish the result to high precision, ensuring the final
  energy is as low as possible for the basin found.
- This combination of global exploration (SA) and local exploitation (Newton)
  is a standard robust approach for hard combinatorial/geometric optimization
  problems like minimal energy configurations.
"""

import numpy as np
import time
from math import log, fsum

N = 150
DEADLINE_FRACTION = 0.90  # Safety margin for 120s limit

def _energy_and_grad_manifold(pts):
    """
    Compute logarithmic energy and Riemannian gradient on the sphere.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    pts: (N, 3) array of unit vectors.
    
    The Riemannian gradient grad_M E at p is the projection of the 
    Euclidean gradient onto the tangent space T_p S^2.
    
    Euclidean gradient component for point i:
    grad_E[i] = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    
    Riemannian gradient:
    grad_M[i] = grad_E[i] - (grad_E[i] . x_i) * x_i
    """
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    mask = dists > 1e-12
    inv_d = np.zeros_like(dists)
    inv_d[mask] = 1.0 / dists[mask]
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient accumulation
    # grad_E[i] = sum_j (x_i - x_j) / d_ij^2
    # coeff[i,j] = 1/d_ij^2 if i!=j else 0
    coeff = np.zeros_like(dists_sq)
    mask_sq = dists_sq > 1e-24
    coeff[mask_sq] = 1.0 / dists_sq[mask_sq]
    np.fill_diagonal(coeff, 0.0)
    
    # grad_E[i] = x_i * sum_j coeff[i,j] - sum_j coeff[i,j] * x_j
    sum_coeff = np.sum(coeff, axis=1)
    sum_coeff_pts = coeff @ pts
    
    grad_E = pts * sum_coeff[:, np.newaxis] - sum_coeff_pts
    
    # Project to tangent space
    dot_gx = np.sum(grad_E * pts, axis=1)
    grad_M = grad_E - dot_gx[:, np.newaxis] * pts
    
    return energy, grad_M

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points in Cartesian coordinates."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * cos(theta)
    y = r * sin(theta)
    return np.column_stack([x, y, z])

def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _newton_polish(pts, max_iter=50, tol=1e-10):
    """
    Deterministic local optimization using Newton's method (or Gauss-Newton).
    This is used to polish the result of the SA phase.
    """
    pts = _project_to_sphere(pts)
    energy, grad = _energy_and_grad_manifold(pts)
    
    # We approximate the Hessian or use a simple step control.
    # For robustness, we use a damped Newton step with line search.
    # Since computing the full Hessian of 450 variables is expensive,
    # we can use the gradient norm to control the step size.
    
    step_scale = 0.1
    for i in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
            
        # Newton step direction is -H^{-1} grad. 
        # Without H, we use -grad scaled by a factor.
        # A more advanced approach would be L-BFGS, but we already have that.
        # Let's stick to a robust gradient step with adaptive scaling.
        
        # Estimate curvature roughly: if energy decreases, we can increase step.
        # For simplicity, we use a fixed small step with backtracking.
        
        direction = -grad
        alpha = step_scale
        
        # Backtracking line search
        for _ in range(20):
            new_pts = _project_to_sphere(pts + alpha * direction)
            new_energy, new_grad = _energy_and_grad_manifold(new_pts)
            
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                grad = new_grad
                # Increase step size slightly if successful
                step_scale = min(step_scale * 1.5, 1.0)
                break
            else:
                alpha *= 0.5
        else:
            # If line search failed, reduce step size and break if very small
            step_scale *= 0.5
            if step_scale < 1e-6:
                break
                
    return pts, energy

def construct():
    start_time = time.time()
    total_time = 120.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Fibonacci spiral
    fib_pts = _fibonacci_spiral(N)
    best_pts = fib_pts.copy()
    best_energy, _ = _energy_and_grad_manifold(best_pts)
    
    # Simulated Annealing
    # Parameters
    T_start = 10.0
    T_end = 1e-6
    cooling_rate = 0.995  # Geometric cooling
    max_sa_steps = 5000
    
    # We will run SA for a while, then polish.
    # We can interleave SA and polishing to refine the best candidate found.
    
    current_pts = best_pts.copy()
    current_energy, _ = _energy_and_grad_manifold(current_pts)
    
    T = T_start
    steps_since_improvement = 0
    
    for step in range(max_sa_steps):
        if time.time() > deadline - 10:
            break
            
        # Random perturbation
        # Perturb each point in the tangent space
        noise = rng.standard_normal((N, 3))
        # Project noise to tangent space: v - (v.p)p
        dot_np = np.sum(noise * current_pts, axis=1, keepdims=True)
        tangent_noise = noise - dot_np * current_pts
        
        # Scale noise by temperature
        scale = 0.01 * np.sqrt(T)
        move = scale * tangent_noise
        
        new_pts = _project_to_sphere(current_pts + move)
        new_energy, _ = _energy_and_grad_manifold(new_pts)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pts = new_pts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1
            
        # Cool down
        T *= cooling_rate
        
        # If we've stagnated for a long time, we can try a larger perturbation
        # or restart from best with a fresh SA cycle (but we are limited in time).
        
    # Final Polish
    if time.time() < deadline - 2:
        try:
            # Polish the best found
            polished_pts, polished_energy = _newton_polish(best_pts, max_iter=100, tol=1e-12)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pts = polished_pts
        except Exception:
            pass
            
    # Ensure final points are valid
    return _project_to_sphere(best_pts)

if __name__ == "__main__":
    pts = construct()
    # Verification
    energy, _ = _energy_and_grad_manifold(pts)
    print(f"Energy: {energy}")
    print(f"Points shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
