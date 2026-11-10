"""Physics-based circle packing optimizer for 200 equal circles in a unit square."""
N = 200
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: must finish well before 150s wall-clock
DEADLINE = time.time() + 140.0  # 10s safety margin


def _energy_and_grad(pos):
    """
    Compute energy and gradient for soft-repulsive potential + wall penalty.
    pos: (N, 2) array of positions in [-0.5, 0.5]^2
    
    Energy = sum_{i<j} 1/d_ij^2 + sum_i wall_penalty_i
    where wall_penalty encourages staying inside [-0.5, 0.5]^2.
    """
    pos = pos.reshape(-1, 2)
    n = pos.shape[0]
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 2)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    
    # Soft repulsive: E_pair = sum_{i<j} 1/d_ij^2
    # Gradient: dE/dd_ij = -2/d_ij^3, dE/dpos_i = 2/d_ij^3 * (pos_i - pos_j)
    # But we use 1/d_sq = 1/d^2, so dE/dpos_i = 2 * (pos_i - pos_j) / d^4
    inv_d4 = 1.0 / (dist_sq ** 2)  # 1/d^4
    np.fill_diagonal(inv_d4, 0.0)
    
    # For pair (i,j): contribution to grad_i is 2*(pos_i - pos_j)/d^4
    # Sum over j for each i:
    # grad_i = 2 * sum_j (pos_i - pos_j) / d_ij^4
    # = 2 * (n*pos_i - sum_j pos_j) / ... no, that's not right
    # Let's compute properly:
    # grad_i = 2 * sum_{j != i} (pos_i - pos_j) / d_ij^4
    # = 2 * (sum_j (pos_i - pos_j) * inv_d4[i,j])
    # = 2 * (pos_i * sum_j inv_d4[i,j] - sum_j pos_j * inv_d4[i,j])
    
    row_sum = np.sum(inv_d4, axis=1)  # (n,)
    # Weighted sum of positions
    weighted_pos = np.sum(inv_d4[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)  # (n, 2)
    
    pair_grad = 2.0 * (pos * row_sum[:, np.newaxis] - weighted_pos)
    
    # Wall penalty: for each coordinate, penalize if outside [-0.5, 0.5]
    # Use smooth penalty: for x > 0.5, penalty = (x - 0.5)^2 * K
    # for x < -0.5, penalty = (-0.5 - x)^2 * K
    K = 100.0  # wall penalty strength
    
    # Compute wall violations
    x = pos[:, 0]
    y = pos[:, 1]
    
    # Right wall: x > 0.5
    vr = np.maximum(x - 0.5, 0.0)
    # Left wall: x < -0.5
    vl = np.maximum(-0.5 - x, 0.0)
    # Top wall: y > 0.5
    vt = np.maximum(y - 0.5, 0.0)
    # Bottom wall: y < -0.5
    vb = np.maximum(-0.5 - y, 0.0)
    
    wall_energy = K * (np.sum(vr**2) + np.sum(vl**2) + np.sum(vt**2) + np.sum(vb**2))
    
    # Wall gradient
    wall_grad = np.zeros_like(pos)
    wall_grad[:, 0] += 2.0 * K * vr
    wall_grad[:, 0] -= 2.0 * K * vl
    wall_grad[:, 1] += 2.0 * K * vt
    wall_grad[:, 1] -= 2.0 * K * vb
    
    total_energy = 0.5 * np.sum(np.sum(inv_d4)) + wall_energy  # 0.5 to avoid double counting
    total_grad = pair_grad + wall_grad
    
    return total_energy, total_grad.flatten()


def _compute_radius(pos):
    """Compute the maximum common radius for given positions."""
    pos = pos.reshape(-1, 2)
    n = pos.shape[0]
    
    # Min distance between any pair
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    # Min distance to wall
    x = pos[:, 0]
    y = pos[:, 1]
    min_x = np.min(0.5 - np.abs(x))
    min_y = np.min(0.5 - np.abs(y))
    min_wall = min(min_x, min_y)
    
    return min(min_dist / 2.0, min_wall)


def construct():
    global DEADLINE
    start_time = time.time()
    
    # Initial configuration: slightly perturbed grid
    m = int(np.ceil(np.sqrt(N)))
    # Use a grid that fits N points
    # For N=200, m=15 gives 225 spots, we use 200
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    pos = grid + 0.01 * np.random.randn(N, 2)
    
    # Clip to stay in bounds initially
    pos = np.clip(pos, -0.49, 0.49)
    
    best_pos = pos.copy()
    best_radius = _compute_radius(best_pos)
    
    # Try multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 20:
            break
        
        if restart == 0:
            current_pos = pos.copy()
        else:
            # Different random perturbations
            np.random.seed(42 + restart * 1000)
            current_pos = grid + 0.015 * np.random.randn(N, 2)
            current_pos = np.clip(current_pos, -0.49, 0.49)
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                x0=current_pos.flatten(),
                jac=lambda x: _energy_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                new_pos = result.x.reshape(-1, 2)
                # Ensure positions are within bounds (clip slightly inside)
                new_pos = np.clip(new_pos, -0.5 + 1e-9, 0.5 - 1e-9)
                new_radius = _compute_radius(new_pos)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_pos = new_pos.copy()
                    
        except Exception:
            pass
        
        # Check time
        if time.time() > DEADLINE - 30:
            break
    
    # Final verification and cleanup
    best_pos = np.clip(best_pos, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_pos
