"""
Strategy: Multi-scale L-BFGS-B with Adaptive Random Restarts.
Instead of Simulated Annealing, we use a robust local optimizer (L-BFGS-B) 
with a smart restart strategy. The key insight is that L-BFGS-B is very fast 
for this problem when given a good initial guess, but it gets stuck in local 
minima. We overcome this by:
1. Starting with diverse initial configurations (Fibonacci, random, quasi-random).
2. Running L-BFGS-B to convergence or near-convergence.
3. If the improvement in the last few iterations is small, or if we hit a 
   maximum iteration count without reaching tight tolerance, we "kick" the 
   configuration with a small random perturbation and restart L-BFGS-B.
4. We manage a global time budget, allowing more restarts if the optimizer 
   is fast, and fewer if it is slow.
5. We use a "coarse-to-fine" approach: initial kicks are larger to escape 
   deep local minima, then smaller for final polishing.

This approach is more deterministic and efficient than SA for finding the 
bottom of the basin once in it, and the restart mechanism provides the 
exploration capability.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s (evaluator kills at 150s, but let's be safe)

def _log_energy_fast(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|).
    Returns energy only.
    """
    n = len(pts)
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf')
    terms = -0.5 * np.log(d_sq)
    energy = math.fsum(terms.tolist())
    return energy

def _objective_and_grad(y_flat):
    """
    Objective function for L-BFGS-B.
    Input: y_flat (3N,)
    Output: energy, gradient (3N,)
    """
    n = len(y_flat) // 3
    y = y_flat.reshape(n, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = y / norms
    
    # Compute energy
    energy = _log_energy_fast(pts)
    
    if energy == float('inf'):
        return energy, np.zeros_like(y_flat)
        
    # Gradient calculation for the normalized points (tangent space)
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq_full = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq_full, 1.0)
    weights = 1.0 / dist_sq_full
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - proj_factor * pts
    
    # Map tangent gradient back to raw space gradient
    # dE/dy_i = grad_tangent_i / ||y_i||
    grad_raw_vec = grad_tangent / norms
    
    return energy, grad_raw_vec.flatten()

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _perturb(pts, magnitude, rng):
    """
    Apply a small random rotation to each point with magnitude 'magnitude'.
    This helps escape local minima.
    """
    new_pts = pts.copy()
    for i in range(len(pts)):
        # Random axis
        axis = rng.normal(size=3)
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis /= norm
        
        # Random angle, scaled by magnitude
        angle = rng.uniform(-magnitude, magnitude)
        
        # Rodrigues rotation
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        new_pts[i] = R @ pts[i]
        
    return new_pts

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Fibonacci
    fib_pts = _fibonacci_sphere(N)
    candidates.append(fib_pts)
    
    # 2. Random (a few different seeds)
    for seed in [42, 123, 999, 2023, 1111]:
        rand_pts = _random_uniform_sphere(N, seed)
        candidates.append(rand_pts)
        
    best_energy = float('inf')
    best_pts = None
    
    # Global RNG for perturbations
    global_rng = np.random.RandomState(42)
    
    # We will iterate through candidates, and for each, run a "kick-and-optimize" loop.
    # The loop continues until time runs out or convergence is reached.
    
    for init_pts in candidates:
        current_pts = init_pts.copy()
        current_energy = _log_energy_fast(current_pts)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
            
        # Kick-and-optimize loop
        # Start with larger kicks, decrease over time
        kick_magnitude = 0.5
        last_energy = current_energy
        stall_count = 0
        
        while time.time() - start_time < DEADLINE - 5:
            # Run L-BFGS-B
            try:
                init_y = current_pts.flatten()
                res = minimize(
                    _objective_and_grad,
                    init_y,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 100,  # Limit iterations per run to allow restarts
                        'maxfun': 100,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'disp': False
                    }
                )
                
                if res.success or res.status == 0:
                    # Converged
                    new_pts = res.x.reshape(N, 3)
                    new_pts = _normalize(new_pts)
                    new_energy = _log_energy_fast(new_pts)
                    
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pts = new_pts.copy()
                    
                    # If converged, we can try a smaller kick to see if we can improve further
                    # Or break if we are very close to optimal? Hard to know.
                    # Let's try a small kick.
                    kick_magnitude *= 0.5
                    if kick_magnitude < 1e-4:
                        # Try a few more small kicks
                        kick_magnitude = 1e-4
                        stall_count += 1
                        if stall_count > 5:
                            break # Give up on this candidate
                    else:
                        stall_count = 0
                        
                else:
                    # Did not converge, take the best point found
                    new_pts = res.x.reshape(N, 3)
                    new_pts = _normalize(new_pts)
                    new_energy = _log_energy_fast(new_pts)
                    
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pts = new_pts.copy()
                    
                    # Kick to escape
                    pass
                
                current_pts = new_pts
                current_energy = new_energy
                
                # Check for stall
                if current_energy > last_energy - 1e-6:
                    stall_count += 1
                else:
                    stall_count = 0
                last_energy = current_energy
                
                # Apply kick
                current_pts = _perturb(current_pts, kick_magnitude, global_rng)
                
            except Exception:
                break

    # Final polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        try:
            init_y = best_pts.flatten()
            res = minimize(
                _objective_and_grad,
                init_y,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'maxfun': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            y_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = y_opt / norms
            
            e, _ = _objective_and_grad(pts_opt.flatten())
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = pts_opt
        except Exception:
            pass

    # Fallback if something went wrong
    if best_pts is None:
        best_pts = fib_pts

    # Final normalization and check
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
