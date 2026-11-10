"""
Strategy: Global Rotation Search + Local L-BFGS Polish.
Instead of moving individual points via stochastic perturbations (Simulated Annealing),
we treat the initial Fibonacci lattice as a rigid body and search for the optimal
global orientation (Euler angles) using a deterministic Nelder-Mead simplex method.
This explores the space of global symmetries and large-scale rearrangements
without breaking the local structure of the lattice, often finding better
basins of attraction for the final local optimization than random point jumps.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 100

# EVOLVE-BLOCK-START

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dist[mask]
    energy = -np.sum(np.log(valid_dists))
    return energy

def _energy_grad(pts):
    """Compute gradient of logarithmic energy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    grad_terms = diff / dist_sq[..., np.newaxis]
    grad = np.sum(grad_terms, axis=1)
    return grad

def _project_tangent(pts, vec):
    """Project vector onto tangent space of sphere at each point."""
    dot = np.sum(pts * vec, axis=1, keepdims=True)
    return vec - dot * pts

def _apply_rotation(pts, angles):
    """
    Apply a sequence of rotations defined by 3 Euler angles (Z-Y-Z convention)
    to the points.
    angles: (alpha, beta, gamma)
    """
    alpha, beta, gamma = angles
    
    # Rotation matrices
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        
    def rot_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
    # Z-Y-Z convention: R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    R = rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)
    
    return pts @ R.T

def _fibonacci_sphere(n):
    """Generate a Fibonacci lattice on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_optimize(pts):
    """
    Perform a high-precision L-BFGS optimization on the tangent space.
    This is the 'polish' step.
    """
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        p = p / norms
        e = _energy(p)
        g = _energy_grad(p)
        g_tan = _project_tangent(p, g)
        return e, g_tan.flatten()
    
    res = minimize(
        fun_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False, 'maxls': 20}
    )
    
    best_pts = res.x.reshape(-1, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    return best_pts, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Base structure: Fibonacci lattice
    base_pts = _fibonacci_sphere(N)
    
    best_pts = base_pts.copy()
    best_energy = _energy(base_pts)
    
    # We will search over the 3 Euler angles to find the best global orientation.
    # Since the energy landscape over rotations is likely complex, we use Nelder-Mead.
    # We also consider a few different "phases" of the Fibonacci lattice (slight shifts)
    # to break symmetries, but primarily we rely on the rotation search.
    
    # Initial simplex for Nelder-Mead
    # Angles are in [0, 2*pi) for alpha, gamma, and [0, pi] for beta
    # We use a small perturbation around the identity rotation (0,0,0)
    # and also try some random starting points for the angles to ensure global coverage.
    
    rng = np.random.RandomState(123)
    
    # Define the objective function for the global search
    def global_objective(angles):
        rotated_pts = _apply_rotation(base_pts, angles)
        # We do a quick local polish here to map the "rotated lattice" to its local minimum
        # This makes the landscape smoother for the global search, effectively
        # searching over the basins of attraction.
        polished_pts, e = _local_optimize(rotated_pts)
        return e
    
    # To save time, we might not want to do a full L-BFGS for every Nelder-Mead step.
    # However, Nelder-Mead on 3D is fast, and L-BFGS on 100 points is ~0.1s.
    # 100 steps * 0.1s = 10s. This is feasible.
    
    # Strategy:
    # 1. Try a few random initial rotations.
    # 2. For each, run a short Nelder-Mead search on the angles.
    # 3. Keep the best result.
    
    num_global_starts = 10
    for i in range(num_global_starts):
        if time.time() > deadline - 20: # Reserve time for final polish
            break
            
        if i == 0:
            # Start at identity
            x0_angles = np.array([0.0, 0.0, 0.0])
        else:
            # Random start
            x0_angles = rng.uniform(0, 2*np.pi, 3)
            x0_angles[1] = rng.uniform(0, np.pi) # Beta is 0..pi
            
        # Nelder-Mead options
        # 'maxiter' controls the number of function evaluations.
        # Since each eval involves a local polish, we limit it.
        try:
            res_global = minimize(
                global_objective, 
                x0_angles, 
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res_global.success or res_global.fun < best_energy:
                # res_global.fun is the energy after local polish
                # We need to retrieve the points. The objective function didn't return them.
                # We re-compute the points for the best angle found.
                best_angles = res_global.x
                rotated_pts = _apply_rotation(base_pts, best_angles)
                final_pts, final_energy = _local_optimize(rotated_pts)
                
                if np.all(np.isfinite(final_pts)):
                    # Check validity
                    diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
                    dists = np.linalg.norm(diff, axis=-1)
                    np.fill_diagonal(dists, 1.0)
                    if np.min(dists) > 1e-9:
                        if final_energy < best_energy:
                            best_energy = final_energy
                            best_pts = final_pts.copy()
                            
        except Exception:
            continue
            
    # Final check and return
    if not np.all(np.isfinite(best_pts)):
        best_pts = base_pts
        
    return best_pts

# EVOLVE-BLOCK-END
