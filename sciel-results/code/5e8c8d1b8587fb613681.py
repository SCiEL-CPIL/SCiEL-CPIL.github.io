"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _score(pos):
    """Compute the packing radius for a set of positions."""
    # pos shape (N, 3)
    if pos.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pos)):
        return 0.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max()
    if wall <= 0:
        return 0.0
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(pos, 'euclidean')
    min_d = d.min()
    return min(wall, min_d / 2.0)


def _energy(pos, r_target):
    """
    Energy function for optimization.
    We want to maximize the minimum of (pairwise_dist/2, wall_dist).
    Equivalent to minimizing violations of constraints:
      - pairwise_dist < 2*r_target
      - wall_dist < r_target
    """
    # Wall violations
    wall_viol = np.maximum(0, r_target - (0.5 - np.abs(pos)))
    # Pairwise violations
    from scipy.spatial.distance import pdist
    d = pdist(pos, 'euclidean')
    pair_viol = np.maximum(0, 2 * r_target - d)
    return np.sum(wall_viol ** 2) + np.sum(pair_viol ** 2)


def _grad_energy(pos, r_target):
    """
    Compute gradient of the energy function.
    """
    grad = np.zeros_like(pos)
    
    # Wall gradient
    abs_pos = np.abs(pos)
    wall_margin = 0.5 - abs_pos
    # For each coordinate, if wall_margin < r_target, gradient pushes inward
    for i in range(N):
        for k in range(3):
            if wall_margin[i, k] < r_target:
                # Push away from the wall
                sign = np.sign(pos[i, k])
                if sign == 0:
                    sign = 1
                grad[i, k] += 2 * (r_target - wall_margin[i, k]) * sign
    
    # Pairwise gradient
    from scipy.spatial.distance import pdist
    d = pdist(pos, 'euclidean')
    # For each pair with d < 2*r_target, push apart
    # pdist returns condensed form; we need to map back
    # For efficiency with N=100, we can do a double loop
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 2 * r_target:
                if dist < 1e-12:
                    # Random perturbation
                    diff = np.random.randn(3) * 1e-8
                    dist = np.linalg.norm(diff)
                grad_force = 2 * (2 * r_target - dist) * diff / dist
                grad[i] += grad_force
                grad[j] -= grad_force
    
    return grad


def _sa_optimize(pos_init, r_target, max_steps=20000, time_limit=10.0):
    """
    Simulated annealing to minimize energy for a fixed r_target.
    """
    start_time = time.time()
    pos = pos_init.copy()
    current_energy = _energy(pos, r_target)
    best_energy = current_energy
    best_pos = pos.copy()
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move
        delta = np.random.randn(N, 3) * 0.01
        new_pos = pos + delta
        
        # Clip to stay within reasonable bounds
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_energy = _energy(new_pos, r_target)
        
        if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy) / T):
            pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pos, best_energy


def _bisection_search(pos_init, time_budget=30.0):
    """
    Use bisection on r_target to find the maximum feasible radius.
    """
    start_time = time.time()
    
    # Initial bounds
    r_low = 0.0
    r_high = 0.15  # Upper bound
    
    # First, check if r_high is feasible
    pos, energy = _sa_optimize(pos_init, r_high, max_steps=5000, time_limit=5.0)
    if energy < 1e-8:
        # r_high is feasible, try higher
        r_low = r_high
        r_high = 0.16
    else:
        # r_high is not feasible, try lower
        r_high = 0.12
    
    # Bisection
    best_pos = pos_init.copy()
    best_r = 0.0
    
    for _ in range(8):  # 8 bisection steps
        if time.time() - start_time > time_budget:
            break
        
        r_mid = (r_low + r_high) / 2.0
        pos, energy = _sa_optimize(pos_init, r_mid, max_steps=10000, time_limit=8.0)
        
        if energy < 1e-8:
            r_low = r_mid
            best_pos = pos
            best_r = r_mid
        else:
            r_high = r_mid
    
    # Final verification
    final_r = _score(best_pos)
    return best_pos, final_r


def _local_optimization(pos, max_iters=500, time_limit=5.0):
    """
    Local optimization using gradient descent to maximize the minimum radius.
    """
    start_time = time.time()
    pos = pos.copy()
    current_r = _score(pos)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the critical constraints
        abs_pos = np.abs(pos)
        wall_margins = 0.5 - abs_pos
        min_wall_idx = np.argmin(wall_margins)
        min_wall_val = wall_margins.min()
        
        from scipy.spatial.distance import pdist
        d = pdist(pos, 'euclidean')
        min_pair_idx = np.argmin(d)
        min_pair_val = d.min() / 2.0
        
        r = min(min_wall_val, min_pair_val)
        
        if r <= 0:
            break
        
        # Compute gradient of r (approximately)
        # This is complex, so we use a simpler approach:
        # Move particles slightly to increase the minimum constraint
        
        # Find which constraints are active
        active_wall = np.where(np.abs(wall_margins - min_wall_val) < 1e-6)[0]
        active_pair = np.where(np.abs(d - min_pair_val * 2) < 1e-6)[0]
        
        # Simple perturbation: move particles slightly
        step_size = 0.001
        delta = np.random.randn(N, 3) * step_size
        
        # For wall-active particles, push inward
        for idx in active_wall:
            i = idx // 3
            k = idx % 3
            sign = np.sign(pos[i, k])
            if sign == 0:
                sign = 1
            delta[i, k] -= sign * step_size
        
        new_pos = pos + delta
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_r = _score(new_pos)
        
        if new_r > current_r:
            pos = new_pos
            current_r = new_r
    
    return pos, current_r


def construct():
    """
    Construct the optimal packing of 100 spheres in a unit cube.
    """
    global_start = time.time()
    deadline = 120.0  # Total time limit
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos_init = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    pos_init += np.random.randn(N, 3) * 0.001
    
    # Multi-restart approach
    best_pos = pos_init
    best_r = _score(pos_init)
    
    # Phase 1: Coarse bisection with SA
    remaining_time = deadline - (time.time() - global_start) - 20.0  # Reserve 20s for final
    if remaining_time > 10:
        pos, r = _bisection_search(pos_init, time_budget=min(remaining_time, 40.0))
        if r > best_r:
            best_pos = pos
            best_r = r
    
    # Phase 2: Local optimization
    remaining_time = deadline - (time.time() - global_start) - 10.0
    if remaining_time > 5:
        pos, r = _local_optimization(best_pos, max_iters=200, time_limit=min(remaining_time, 10.0))
        if r > best_r:
            best_pos = pos
            best_r = r
    
    # Phase 3: Additional SA restarts if time permits
    remaining_time = deadline - (time.time() - global_start) - 5.0
    if remaining_time > 15:
        # Try a few more restarts with different initial conditions
        for restart in range(3):
            if time.time() - global_start > deadline - 5.0:
                break
            
            # Perturb the best solution
            pos_perturbed = best_pos + np.random.randn(N, 3) * 0.05
            pos_perturbed = np.clip(pos_perturbed, -0.5, 0.5)
            
            # Try to improve with SA at current best radius
            r_target = best_r * 1.01
            pos, energy = _sa_optimize(pos_perturbed, r_target, max_steps=5000, time_limit=min(remaining_time / 3, 8.0))
            
            if energy < 1e-8:
                r = _score(pos)
                if r > best_r:
                    best_pos = pos
                    best_r = r
    
    # Final safety check
    final_r = _score(best_pos)
    if final_r <= 0 or not np.all(np.isfinite(best_pos)):
        # Fallback to grid
        return grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
