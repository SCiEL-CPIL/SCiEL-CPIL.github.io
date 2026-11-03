"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _compute_radius(c):
    """Compute the maximum valid radius for centres c in [-0.5, 0.5]^3."""
    n = len(c)
    if n == 0:
        return 0.0
    # Wall distance
    wall = np.min(0.5 - np.abs(c), axis=1)
    r_wall = np.min(wall)
    if r_wall <= 0:
        return 0.0
    # Pairwise distances
    # Use a vectorized approach for efficiency
    # For 100 points, this is manageable
    d = np.zeros((n, n))
    for i in range(n):
        diff = c[i+1:] - c[i]
        d[i, i+1:] = np.sqrt(np.sum(diff**2, axis=1))
        d[i+1:, i] = d[i, i+1:]
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r_pair = min_pair / 2.0
    return min(r_wall, r_pair)


def _energy_and_grad(c, r_target):
    """
    Energy: sum of violations for pairs with distance < 2*r_target
    and walls with distance < r_target.
    Gradient with respect to c.
    """
    n = len(c)
    grad = np.zeros_like(c)
    energy = 0.0
    
    # Wall constraints: each point must satisfy |c_i| <= 0.5 - r_target in each coord
    # Penalty: for each coord, if c_ik > 0.5 - r_target, push left; if < -(0.5-r_target), push right
    bound = 0.5 - r_target
    for i in range(n):
        for k in range(3):
            val = c[i, k]
            if val > bound:
                excess = val - bound
                energy += 0.5 * excess * excess
                grad[i, k] -= excess
            elif val < -bound:
                excess = -bound - val
                energy += 0.5 * excess * excess
                grad[i, k] += excess
    
    # Pair constraints: for each pair (i,j), if dist < 2*r_target, penalize
    min_dist = 2 * r_target
    for i in range(n):
        for j in range(i+1, n):
            diff = c[i] - c[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < min_dist:
                if dist < 1e-12:
                    # Degenerate: push apart randomly
                    grad[i] += 0.1 * (np.random.rand(3) - 0.5)
                    grad[j] -= 0.1 * (np.random.rand(3) - 0.5)
                    energy += 0.5 * (min_dist - 0) ** 2
                else:
                    excess = min_dist - dist
                    energy += 0.5 * excess * excess
                    # d(dist)/d(c_i) = -diff/dist, d(excess)/d(c_i) = d(min_dist - dist)/d(c_i) = diff/dist
                    # d(0.5*excess^2)/d(c_i) = excess * diff/dist
                    factor = excess / dist
                    grad[i] += factor * diff
                    grad[j] -= factor * diff
    
    return energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initialize with a slightly perturbed grid for better starting point
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].astype(np.float64).copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    c += 0.01 * np.random.randn(N, 3)
    
    # Compute initial radius
    r_current = _compute_radius(c)
    best_c = c.copy()
    best_r = r_current
    
    # Iterative radius increase with energy minimization
    # We'll try to increase the radius in small steps, minimizing energy at each step
    r_target = r_current
    step_size = 0.001  # Initial step in radius
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try to increase radius
        r_target = best_r + step_size
        
        # Check if this is even feasible (rough check)
        # For 100 spheres in unit cube, max radius is about 0.111
        if r_target > 0.115:
            step_size *= 0.5
            continue
        
        # Minimize energy at this target radius
        lr = 0.001  # Learning rate
        for sub_iter in range(200):
            if time.time() > deadline:
                break
            
            energy, grad = _energy_and_grad(c, r_target)
            
            if energy < 1e-10:
                break
            
            # Gradient descent with line search
            c_new = c - lr * grad
            
            # Simple projection to keep in cube
            c_new = np.clip(c_new, -0.5, 0.5)
            
            energy_new, _ = _energy_and_grad(c_new, r_target)
            
            if energy_new < energy:
                c = c_new
            else:
                lr *= 0.5
        
        # After minimization, check if we achieved the target radius
        r_achieved = _compute_radius(c)
        
        if r_achieved >= r_target - 1e-8:
            # Success! Update best
            best_c = c.copy()
            best_r = r_achieved
            step_size = min(step_size * 1.2, 0.005)
        else:
            # Failed, reduce step
            step_size *= 0.5
            c = best_c.copy()
            if step_size < 1e-6:
                break
    
    # Final polish: try to improve the best solution found
    if time.time() < deadline - 5.0:
        c = best_c.copy()
        r_polish = best_r
        # Small local optimization to squeeze out more
        for polish_iter in range(1000):
            if time.time() > deadline:
                break
            
            # Compute gradient of negative radius (we want to maximize)
            # Use subgradient approach
            n = len(c)
            # Find the binding constraints
            wall = np.min(0.5 - np.abs(c), axis=1)
            r_wall = np.min(wall)
            
            # Pairwise
            min_pair = np.inf
            min_pair_idx = (0, 0)
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt(np.sum((c[i] - c[j])**2))
                    if d < min_pair:
                        min_pair = d
                        min_pair_idx = (i, j)
            
            r_pair = min_pair / 2.0
            r_current = min(r_wall, r_pair)
            
            # Determine which constraint is binding
            if r_wall < r_pair:
                # Wall is binding
                # Find which point and coordinate
                min_wall_idx = np.argmin(wall)
                min_wall_coord = np.argmin(0.5 - np.abs(c[min_wall_idx]))
                # Push this point inward
                sign = np.sign(c[min_wall_idx, min_wall_coord])
                if sign == 0:
                    sign = 1
                grad = np.zeros_like(c)
                grad[min_wall_idx, min_wall_coord] = -sign
            else:
                # Pair is binding
                i, j = min_pair_idx
                diff = c[i] - c[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-12:
                    break
                grad = np.zeros_like(c)
                grad[i] += diff / dist
                grad[j] -= diff / dist
            
            # Move in direction that increases radius
            step = 0.001
            c_new = c + step * grad
            c_new = np.clip(c_new, -0.5, 0.5)
            
            r_new = _compute_radius(c_new)
            if r_new > r_polish:
                c = c_new
                r_polish = r_new
            else:
                step *= 0.5
                if step < 1e-8:
                    break
        
        if r_polish > best_r:
            best_c = c.copy()
            best_r = r_polish
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_c
# EVOLVE-BLOCK-END
