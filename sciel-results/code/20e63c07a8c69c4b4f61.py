"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary
    # pos is (N, 2)
    # dist to x=0.5 is 0.5 - x, dist to x=-0.5 is 0.5 + x
    # min dist to boundary is min(0.5 - |x|, 0.5 - |y|)
    dist_x = 0.5 - np.abs(pos[:, 0])
    dist_y = 0.5 - np.abs(pos[:, 1])
    dist_boundary = np.minimum(dist_x, dist_y)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    # Using broadcasting
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = float(np.min(dist_sq))
    min_pair = math.sqrt(min_pair_sq)
    
    return min(min_pair / 2.0, min_boundary)

def _initial_hex_grid(n):
    """Create initial hexagonal lattice packing."""
    # Approximate rows and cols
    # Hex packing density is sqrt(3)/2.
    # Area per point ~ 2/sqrt(3) * r^2.
    # For n points in unit square, r ~ 0.5 / sqrt(n * 2/sqrt(3)) ?
    # Let's just generate a grid and scale.
    
    rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    cols = int(np.ceil(n / rows))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = col
            y = row * np.sqrt(3) / 2
            if row % 2 == 1:
                x += 0.5
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n])
    
    # Center
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.48, 0.48]
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.48 / max_coord
        points *= scale
        
    return points

def _soft_objective_and_grad(pos_flat, lam):
    """
    Objective: Sum of exp(-lam * d^2) for pairs and boundaries.
    We minimize this.
    """
    pos = pos_flat.reshape(-1, 2)
    n = len(pos)
    
    # Pairwise terms
    # d_ij^2
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # exp term
    exp_term = np.exp(-lam * dist_sq)
    exp_term[~mask] = 0.0
    
    obj = np.sum(exp_term)
    
    # Gradient w.r.t pos_i
    # d/dx_i exp(-lam * d_ij^2) = -2*lam*(x_i - x_j)*exp(-lam*d_ij^2)
    # Sum over j
    # grad_i = -2 * lam * sum_j (pos_i - pos_j) * exp_term_ij
    
    # (pos_i - pos_j) is diff[i,j,:]
    # We need sum over j for each i
    # grad[i] = -2 * lam * sum_j (diff[i,j] * exp_term[i,j])
    
    # Vectorized sum
    # diff is (N, N, 2), exp_term is (N, N)
    # We want (N, 2)
    # Multiply diff by exp_term[:, :, np.newaxis]
    weighted_diff = diff * exp_term[:, :, np.newaxis]
    grad_pairs = -2.0 * lam * np.sum(weighted_diff, axis=1)
    
    # Boundary terms
    # Penalize closeness to boundaries
    # Term: exp(-lam * (0.5 - |x|)^2) ? No, that's not smooth at 0.
    # Use: exp(-lam * (0.5 - x)^2) + exp(-lam * (0.5 + x)^2)
    # This pushes x away from 0.5 and -0.5.
    
    x = pos[:, 0]
    y = pos[:, 1]
    
    # Objective components
    obj_x_right = np.exp(-lam * (0.5 - x)**2)
    obj_x_left = np.exp(-lam * (0.5 + x)**2)
    obj_y_right = np.exp(-lam * (0.5 - y)**2)
    obj_y_left = np.exp(-lam * (0.5 + y)**2)
    
    obj += np.sum(obj_x_right) + np.sum(obj_x_left) + np.sum(obj_y_right) + np.sum(obj_y_left)
    
    # Gradients
    # d/dx exp(-lam * (0.5 - x)^2) = -2*lam*(0.5 - x)*(-1) * exp(...) = 2*lam*(0.5 - x)*exp(...)
    # d/dx exp(-lam * (0.5 + x)^2) = -2*lam*(0.5 + x)*(1) * exp(...) = -2*lam*(0.5 + x)*exp(...)
    
    grad_x = 2.0 * lam * (0.5 - x) * obj_x_right - 2.0 * lam * (0.5 + x) * obj_x_left
    grad_y = 2.0 * lam * (0.5 - y) * obj_y_right - 2.0 * lam * (0.5 + y) * obj_y_left
    
    grad = np.zeros_like(pos)
    grad[:, 0] += grad_x
    grad[:, 1] += grad_y
    
    total_grad = grad_pairs + grad
    
    # Flatten gradient
    return obj, total_grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Initial positions
    init_pos = _initial_hex_grid(N)
    # Add small noise to break symmetry
    rng = np.random.RandomState(42)
    init_pos += rng.uniform(-0.01, 0.01, size=init_pos.shape)
    init_pos = np.clip(init_pos, -0.49, 0.49)
    
    best_pos = init_pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Annealing the lambda parameter
    # Start with low lambda (smooth), increase to high (sharp)
    lambdas = [10, 20, 50, 100, 200, 400]
    
    current_pos = init_pos.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 20:
            break
            
        pos_flat = current_pos.flatten()
        
        def objective_and_grad(x):
            obj, grad = _soft_objective_and_grad(x, lam)
            return obj, grad
        
        try:
            # Use L-BFGS-B as it handles bounds well and is efficient
            # Bounds: -0.4999 to 0.4999
            bounds = [(-0.4999, 0.4999)] * (2 * N)
            
            result = minimize(
                objective_and_grad,
                pos_flat,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 500}
            )
            
            if result.success:
                new_pos = result.x.reshape(-1, 2)
                new_pos = np.clip(new_pos, -0.4999, 0.4999)
                new_r = _compute_radius(new_pos)
                
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
                
                # Update current position for next iteration
                current_pos = new_pos.copy()
                
        except Exception as e:
            pass
    
    # Final local refinement using exact radius
    # Try to nudge the limiting points
    if time.time() < deadline - 5 and best_pos is not None:
        pos = best_pos.copy()
        current_r = best_r
        
        # Identify limiting pairs
        # We'll just do a few iterations of "move point i in direction of gradient of min distance"
        # This is tricky because min distance is non-differentiable.
        # Instead, we can use a simple random local search with small steps.
        
        step_size = 0.001
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            
            # Pick a random point
            i = np.random.randint(N)
            # Perturb it
            delta = np.random.randn(2) * step_size
            new_pos = pos.copy()
            new_pos[i] += delta
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            
            new_r = _compute_radius(new_pos)
            
            if new_r > current_r:
                pos = new_pos
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = pos.copy()
            else:
                # Reduce step size if stuck?
                step_size *= 0.9
                if step_size < 1e-6:
                    break
    
    return best_pos
# EVOLVE-BLOCK-END
