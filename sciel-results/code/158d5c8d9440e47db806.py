"""Rewritten program: velocity-Verlet rigid-body dynamics on the sphere for log-energy minimization."""
import numpy as np
import time
import math

N = 150

def _energy_and_grad(pts):
    """Compute log-energy and gradient for points on unit sphere (chordal distances).
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  (for -log(r) potential)
    Wait: potential is -log(r), so force is -d(-log(r))/dr * r_hat = (1/r) r_hat
    Actually let's be careful.
    
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i, j>i} -1/|x_i-x_j| * (x_i-x_j)/|x_i-x_j| + sum_{j<i} -1/|x_j-x_i| * (x_i-x_j)/|x_j-x_i|
    
    More carefully: for pair (i,j), contribution to E is -log(r_ij) where r_ij = |x_i - x_j|
    dE/dx_i from pair (i,j) = -1/r_ij * (x_i - x_j)/r_ij = -(x_i - x_j)/r_ij^2
    dE/dx_j from pair (i,j) = -1/r_ij * (x_j - x_i)/r_ij = (x_i - x_j)/r_ij^2
    
    So grad_i = sum_{j != i} -(x_i - x_j) / |x_i - x_j|^2
    
    Let me verify: if x_i moves toward x_j, r decreases, -log(r) increases, so gradient should push x_i away from x_j.
    -(x_i - x_j)/r^2 points from x_i toward x_j... that's wrong sign.
    
    Let me redo: E_pair = -log(r), r = |x_i - x_j|
    dE_pair/dx_i = -d(r)/dx_i / r = -(x_i - x_j)/r / r = -(x_i - x_j)/r^2
    
    If x_i moves in direction of x_j (i.e., x_i - x_j decreases in magnitude), then (x_i-x_j) points from j to i.
    Moving x_i toward x_j means moving in direction -(x_i-x_j)/r.
    dE/dx_i dotted with that direction: [-(x_i-x_j)/r^2] . [-(x_i-x_j)/r] = |x_i-x_j|^2/r^3 = 1/r > 0.
    So energy increases when moving toward each other, which makes sense (repulsive).
    The gradient points in direction of increasing energy, so to minimize we go opposite to gradient.
    
    So grad_i = sum_{j != i} -(x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    # Compute pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # shape (n, n, 3)
    r2 = np.sum(diff * diff, axis=2)  # shape (n, n)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    
    # Energy: sum over i<j of -log(r_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(r[iu]))
    
    # Gradient: grad_i = sum_{j != i} -(x_i - x_j) / r_ij^2
    # = -sum_{j != i} diff[i,j] / r2[i,j]
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    grad = -np.sum(diff * inv_r2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # Initial configuration: Fibonacci sphere
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # Compute initial energy
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    # Velocity-Verlet with damping
    vel = np.zeros_like(pts)
    dt = 0.01
    damping = 0.999  # slight damping to dissipate energy
    
    # Adaptive timestep parameters
    min_dt = 1e-6
    max_dt = 0.05
    
    energy, grad = _energy_and_grad(pts)
    
    iteration = 0
    no_improve_count = 0
    best_energy_so_far = energy
    
    while time.time() < deadline:
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        
        # Update velocity (half step)
        vel = vel * damping + 0.5 * dt * (-grad)  # negative gradient = force direction for minimization
        
        # Project velocity to tangent space (remove radial component)
        vel = vel - np.sum(vel * pts, axis=1, keepdims=True) * pts
        
        # Update position
        pts_new = pts + dt * vel
        pts_new = _project_to_sphere(pts_new)
        
        # Compute new energy and gradient
        energy_new, grad_new = _energy_and_grad(pts_new)
        
        # Adaptive timestep: reduce if energy increased too much or gradient is large
        if not np.isfinite(energy_new) or energy_new > best_energy_so_far + 1.0:
            dt = max(min_dt, dt * 0.5)
            continue
        
        # Accept step
        pts = pts_new
        energy = energy_new
        grad = grad_new
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Adapt timestep
        if grad_norm > 10.0:
            dt = max(min_dt, dt * 0.7)
        elif grad_norm < 0.01 and dt < max_dt:
            dt = min(max_dt, dt * 1.1)
        
        # If no improvement for many steps, try to escape local minimum
        if no_improve_count > 500:
            # Add small random perturbation
            pts = pts + 0.01 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
            vel = np.zeros_like(pts)
            dt = min_dt * 10
            no_improve_count = 0
            energy, grad = _energy_and_grad(pts)
        
        iteration += 1
    
    # Final polish: use gradient descent with smaller steps
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        pts = best_pts.copy()
        vel = np.zeros_like(pts)
        dt = 1e-4
        energy, grad = _energy_and_grad(pts)
        
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            
            # Gradient descent step with momentum
            vel = 0.9 * vel + 0.1 * (-grad)
            # Project to tangent space
            vel = vel - np.sum(vel * pts, axis=1, keepdims=True) * pts
            
            pts_new = pts + dt * vel
            pts_new = _project_to_sphere(pts_new)
            
            energy_new, grad_new = _energy_and_grad(pts_new)
            
            if np.isfinite(energy_new) and energy_new < energy:
                pts = pts_new
                energy = energy_new
                grad = grad_new
                if energy < best_energy:
                    best_energy = energy
                    best_pts = pts.copy()
            else:
                dt = max(1e-6, dt * 0.5)
        
        # Final check
        energy_final, _ = _energy_and_grad(best_pts)
        if energy_final < best_energy:
            best_energy = energy_final
    
    # Ensure all points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts
