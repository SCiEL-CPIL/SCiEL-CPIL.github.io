"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 150.0  # seconds


def _compute_radius(c):
    """Compute the maximum common radius for centres c (N,3)."""
    c = np.asarray(c)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Wall constraint
    wall = 0.5 - np.abs(c)
    r_wall = np.min(wall)
    # Pairwise distances
    # Use broadcasting for N=100: 100x100 matrix
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_wall, r_pair)


def _initial_config():
    """Generate a good initial configuration using a BCC-like lattice."""
    # Use a 5x5x4 grid with BCC offset for some points
    # Actually, let's use a simple approach: 5x5x4 = 100 points exactly
    # But we need them to be well-spread. Let's use a perturbed grid.
    
    # Start with a 5x5x4 grid
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.4, 0.4, 4)
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    
    points = np.array(points)
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    points += np.random.uniform(-0.02, 0.02, size=points.shape)
    
    # Clip to stay inside
    points = np.clip(points, -0.45, 0.45)
    
    return points


def _violation_obj(c, target_r):
    """
    Objective: sum of violations of constraints.
    For each pair (i,j): violation = max(0, 2*target_r - |c_i - c_j|)
    For each point i and each wall: violation = max(0, target_r - (0.5 - |c_ik|))
    We minimize the sum of squared violations.
    """
    c = np.reshape(c, (N, 3))
    if not np.all(np.isfinite(c)):
        return 1e10
    
    # Wall violations
    wall_dist = 0.5 - np.abs(c)  # (N, 3)
    wall_viol = np.maximum(0, target_r - wall_dist)
    wall_obj = np.sum(wall_viol ** 2)
    
    # Pairwise violations
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0, 2 * target_r - dist)
    pair_obj = np.sum(pair_viol ** 2)
    
    return wall_obj + pair_obj


def _violation_grad(c, target_r):
    """Gradient of the violation objective."""
    c = np.reshape(c, (N, 3))
    grad = np.zeros_like(c)
    
    # Wall gradient
    wall_dist = 0.5 - np.abs(c)
    wall_viol = np.maximum(0, target_r - wall_dist)
    # d/dc of (target_r - (0.5 - |c|))^2 when violated
    # = 2*(target_r - wall_dist) * sign(c)
    grad_wall = 2 * wall_viol * np.sign(c)
    grad += grad_wall
    
    # Pairwise gradient
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0, 2 * target_r - dist)
    
    # For each pair (i,j) with violation, gradient contribution to c_i:
    # d/dc_i of (2*r - |c_i - c_j|)^2 = 2*(2*r - dist) * (c_i - c_j)/dist
    # Sum over j
    # Use vectorized form
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d < 1e-12:
                continue
            v = pair_viol[i, j]
            if v > 0:
                grad[i] += 2 * v * (c[i] - c[j]) / d
    
    return grad


def _optimize(c0, target_r, max_iter=500, timeout=100.0):
    """Run BFGS to minimize violations for a given target radius."""
    x0 = c0.flatten()
    
    def obj(x):
        return _violation_obj(x, target_r)
    
    def grad(x):
        return _violation_grad(x, target_r).flatten()
    
    result = minimize(
        obj, x0, jac=grad, method='BFGS',
        options={'maxiter': max_iter, 'gtol': 1e-10}
    )
    
    return result.x.reshape(N, 3)


def _squeeze(c, iterations=200, step_scale=0.01):
    """
    Final high-precision polish: iteratively try to increase radius
    by pushing points apart and away from walls.
    """
    r = _compute_radius(c)
    
    for it in range(iterations):
        # Compute current radius
        r = _compute_radius(c)
        
        # Try to increase radius slightly
        new_r = r * (1 + 1e-8)
        
        # Compute forces: push points apart and away from walls
        forces = np.zeros_like(c)
        
        # Wall forces
        for i in range(N):
            for k in range(3):
                wall_d = 0.5 - abs(c[i, k])
                if wall_d < new_r:
                    # Push away from wall
                    forces[i, k] += 0.1 * (new_r - wall_d) * np.sign(c[i, k])
        
        # Pairwise forces
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < 2 * new_r:
                    if d < 1e-12:
                        continue
                    f = 0.1 * (2 * new_r - d) / d
                    forces[i] += f * (c[i] - c[j])
                    forces[j] -= f * (c[i] - c[j])
        
        # Apply forces with adaptive step size
        step = step_scale / (1 + it * 0.05)
        c_new = c + step * forces
        
        # Check if new configuration has larger or equal radius
        r_new = _compute_radius(c_new)
        if r_new >= r - 1e-15:
            c = c_new
        else:
            # Try smaller step
            step /= 10
            c_new = c + step * forces
            r_new = _compute_radius(c_new)
            if r_new >= r - 1e-15:
                c = c_new
    
    return c


def construct():
    start_time = time.time()
    
    # Initial configuration
    c = _initial_config()
    
    # Get initial radius
    r = _compute_radius(c)
    
    # Iteratively increase target radius and minimize violations
    # Start from current radius and gradually increase
    target_r = r
    best_c = c.copy()
    best_r = r
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 15:  # Safety margin
            break
        
        # Try to increase target radius
        target_r *= 1.005  # 0.5% increase
        
        # Check if target is feasible (current config already satisfies?)
        current_r = _compute_radius(c)
        if target_r <= current_r:
            # Already feasible, try bigger jump
            target_r = current_r * 1.01
            continue
        
        # Run BFGS to minimize violations
        max_iter = 300
        if elapsed > _DEADLINE - 40:
            max_iter = 100
        
        c_opt = _optimize(c, target_r, max_iter=max_iter)
        
        # Check the resulting radius
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_c = c_opt.copy()
            best_r = r_opt
            c = c_opt
        else:
            # If we couldn't improve, try a smaller increase
            target_r = r * 1.001
            c_opt = _optimize(c, target_r, max_iter=100)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_c = c_opt.copy()
                best_r = r_opt
                c = c_opt
            else:
                # Can't improve further with this approach, break
                break
        
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 30:
            break
    
    # Final squeeze pass
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        squeeze_iters = max(50, int((_DEADLINE - 10 - elapsed) * 10))
        best_c = _squeeze(best_c, iterations=squeeze_iters)
        best_r = _compute_radius(best_c)
    
    # Ensure validity: clip to cube
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
