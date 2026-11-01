import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    E = sum_{i<j} log(1/||x_i - x_j||)
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # points shape (N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Gradient calculation
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy calculation
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    return energy, grad

def _objective_and_grad(x):
    """
    Objective function and gradient for scipy.minimize.
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points_sphere = points / norms
    
    e, g = _energy_and_grad(points_sphere)
    
    # Penalty for leaving sphere
    lambda_pen = 100.0
    diff_pen = points - points_sphere
    penalty = 0.5 * lambda_pen * np.sum(diff_pen**2)
    grad_pen = lambda_pen * diff_pen
    
    total_grad = (g + grad_pen).flatten()
    return e + penalty, total_grad

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix."""
    # Use QR decomposition of a random normal matrix
    A = rng.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure right-handedness
    d = np.sign(np.linalg.det(Q))
    Q = Q * np.array([1, 1, d])
    return Q

def _sa_step(points, T, rng):
    """
    Perform one Simulated Annealing step.
    Moves a batch of points by a small random rotation.
    Returns new_points, delta_E, E_new
    """
    # Choose batch size
    batch_size = rng.randint(1, 4)
    indices = rng.choice(N, size=batch_size, replace=False)
    
    # Create a copy for the move
    new_points = points.copy()
    
    # Generate a small random rotation
    # Angle proportional to sqrt(T) to allow larger moves at high T
    angle = 0.1 * np.sqrt(T)
    if angle > 0.5:
        angle = 0.5
        
    # Random axis
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    
    # Construct rotation matrix using Rodrigues' formula
    # R = I * cos(theta) + (1-cos(theta)) * (u u^T) + sin(theta) * [u]_x
    u = axis
    u = u[:, np.newaxis]
    K = np.array([[0, -u[2], u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]])
    c = np.cos(angle)
    s = np.sin(angle)
    R = c * np.eye(3) + (1 - c) * (u @ u.T) + s * K
    
    # Apply rotation to the selected points
    # We rotate them around the origin? Or just perturb?
    # Rotating around the origin keeps them on the sphere.
    new_points[indices] = new_points[indices] @ R.T
    
    # Ensure they are still on sphere (numerical drift)
    new_points[indices] = _project_to_sphere(new_points[indices])
    
    # Calculate energy difference
    E_old, _ = _energy_and_grad(points)
    E_new, _ = _energy_and_grad(new_points)
    delta_E = E_new - E_old
    
    return new_points, delta_E, E_new

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # 1. Initialization
    points = _fibonacci_spiral()
    best_points = points.copy()
    e_init, _ = _energy_and_grad(best_points)
    best_energy = e_init
    
    # 2. Initial L-BFGS Polish
    try:
        x0 = best_points.flatten()
        res = minimize(_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20})
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(N, 3))
            e_new, _ = _energy_and_grad(new_points)
            if e_new < best_energy:
                best_energy = e_new
                best_points = new_points
    except Exception:
        pass
    
    # Save pre-SA best
    pre_sa_points = best_points.copy()
    pre_sa_energy = best_energy
    
    # 3. Simulated Annealing Phase
    # Time budget for SA: 90 seconds
    sa_start = time.time()
    sa_time_limit = 90.0
    
    T = 5.0  # Initial temperature
    cooling_rate = 0.999
    current_points = best_points.copy()
    current_energy = best_energy
    
    # We store the best found during SA
    sa_best_points = current_points.copy()
    sa_best_energy = current_energy
    
    while time.time() - sa_start < sa_time_limit:
        # Check for time out in the loop
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
            
        # Perform a move
        new_points, delta_E, E_new = _sa_step(current_points, T, rng)
        
        # Metropolis Criterion
        if delta_E < 0 or (T > 1e-6 and rng.rand() < np.exp(-delta_E / T)):
            current_points = new_points
            current_energy = E_new
            
            if E_new < sa_best_energy:
                sa_best_energy = E_new
                sa_best_points = new_points.copy()
        
        # Cool down
        T *= cooling_rate
        
        # If T is very low, we might want to restart or just stop
        if T < 0.01:
            break

    # 4. Final Polish with L-BFGS on the best SA result
    final_points = sa_best_points.copy()
    
    try:
        x0 = final_points.flatten()
        # High precision polish
        res = minimize(_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(N, 3))
            e_new, _ = _energy_and_grad(new_points)
            if e_new < sa_best_energy:
                sa_best_energy = e_new
                sa_best_points = new_points
    except Exception:
        pass
    
    # Compare pre-SA and post-SA results
    if sa_best_energy < pre_sa_energy:
        best_points = sa_best_points
        best_energy = sa_best_energy
    else:
        best_points = pre_sa_points
        best_energy = pre_sa_energy
        
    # Final Projection and Validation
    best_points = _project_to_sphere(best_points)
    
    # Collision check
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng_coll = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng_coll.randn(3) * 1e-6
                best_points[i] = _project_to_sphere(best_points[i].reshape(1,3))[0]

    return best_points
