"""Physics-based packing: molecular dynamics with velocity Verlet, then local polish."""
import numpy as np
import time
import math

N = 50


def _pairwise_dists(pos):
    """Return (1, N) array of min distance to any other circle, and (1, N) array of min distance to wall."""
    # pos shape (N, 2)
    # pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist, axis=1)  # (N,)

    # wall distances
    wall = np.minimum(0.5 - np.abs(pos[:, 0]), 0.5 - np.abs(pos[:, 1]))  # (N,)

    return min_pair, wall


def _score(pos):
    """Compute the radius (score) for a given configuration."""
    min_pair, wall = _pairwise_dists(pos)
    r = np.minimum(np.min(min_pair) / 2.0, np.min(wall))
    return r


def _md_simulate(pos0, dt=0.001, steps=50000, damping=0.999):
    """
    Molecular dynamics simulation with soft repulsive forces.
    We use a potential that pushes circles apart when they overlap.
    """
    pos = pos0.copy().astype(np.float64)
    vel = np.zeros_like(pos)

    # Potential: for each pair, if distance < 2r_target, apply repulsive force
    # We'll use a spring-like force for overlaps
    # Actually, let's use a simpler approach: force = k * (d_min - dist) if dist < d_min
    # But we don't know the target radius in advance.
    
    # Better approach: use a repulsive potential U(r) = k * max(0, r0 - r)^2
    # where r0 is some reference distance. We can set r0 to be the initial min distance.
    
    # Let's use a different strategy: 
    # 1. Start with a good initial configuration (hexagonal-ish)
    # 2. Use MD with a soft repulsive force to resolve overlaps
    # 3. Then do local optimization
    
    # For MD, we need forces. Let's use:
    # F_ij = k * (r_min - |r_i - r_j|) * (r_i - r_j)/|r_i - r_j| if |r_i - r_j| < r_min
    # where r_min is a small value (like 0.01) to just resolve overlaps
    
    # Actually, a cleaner approach for packing:
    # Use a potential that maximizes the minimum distance.
    # Let's use: U = -sum_{i<j} min(d_ij, d_max) + sum_i min(d_i_wall, d_max)
    # This is not smooth. 
    # 
    # Alternative: use a soft min function.
    # 
    # Let me try a different approach: 
    # 1. Start with hexagonal packing
    # 2. Use simulated annealing or gradient-based optimization
    # 3. The MD approach is tricky because we don't have a natural potential for "maximize min distance"
    
    # Let's use a simpler approach: 
    # Use a repulsive force that is proportional to the overlap, with a target radius.
    # We can estimate the target radius from the initial configuration.
    
    # For now, let's just do a simple relaxation:
    # If two circles overlap (dist < 2*r_est), push them apart.
    
    r_est = 0.07  # initial estimate
    
    k = 100.0  # spring constant
    mass = 1.0
    
    for step in range(steps):
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise forces
        for i in range(N):
            for j in range(i + 1, N):
                diff = pos[i] - pos[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 2 * r_est and dist > 1e-10:
                    # Repulsive force
                    f_mag = k * (2 * r_est - dist)
                    f = f_mag * diff / dist
                    forces[i] += f
                    forces[j] -= f
        
        # Wall forces
        for i in range(N):
            for axis in range(2):
                if pos[i, axis] > 0.5 - r_est:
                    forces[i, axis] -= k * (pos[i, axis] - (0.5 - r_est))
                elif pos[i, axis] < -0.5 + r_est:
                    forces[i, axis] += k * ((-0.5 + r_est) - pos[i, axis])
        
        # Velocity Verlet integration
        vel = (vel + forces / mass * dt) * damping
        pos = pos + vel * dt
        
        # Keep in bounds (hard constraint)
        pos[:, 0] = np.clip(pos[:, 0], -0.5 + r_est, 0.5 - r_est)
        pos[:, 1] = np.clip(pos[:, 1], -0.5 + r_est, 0.5 - r_est)
    
    return pos


def _hexagonal_init(n):
    """Generate a hexagonal close-packed initial configuration."""
    # Hexagonal packing: rows offset by half the horizontal spacing
    # For n circles, we can arrange them in a roughly square grid of hexagons
    
    # Let's use a grid with hexagonal offset
    m = int(math.ceil(math.sqrt(n)))
    
    # Horizontal spacing for hexagonal packing
    # In a hexagonal lattice, the ratio of vertical to horizontal spacing is sqrt(3)/2
    # Let's use a spacing that fits in the unit square
    
    # Try different numbers of columns
    best_pos = None
    best_score = -1
    
    for cols in range(5, 11):
        rows = int(math.ceil(n / cols))
        
        # Horizontal spacing
        dx = 1.0 / (cols + 1)
        # Vertical spacing for hexagonal packing
        dy = dx * math.sqrt(3) / 2
        
        # Check if it fits
        if rows * dy > 1.0:
            continue
        
        pos = []
        count = 0
        for row in range(rows):
            if count >= n:
                break
            # Offset alternating rows
            offset = 0.5 * dx if row % 2 == 1 else 0.0
            for col in range(cols):
                if count >= n:
                    break
                x = -0.5 + dx * (col + 1) + offset
                y = -0.5 + dy * (row + 0.5)
                
                # Keep in bounds
                x = np.clip(x, -0.5 + 0.07, 0.5 - 0.07)
                y = np.clip(y, -0.5 + 0.07, 0.5 - 0.07)
                
                pos.append([x, y])
                count += 1
        
        if len(pos) == n:
            pos = np.array(pos)
            score = _score(pos)
            if score > best_score:
                best_score = score
                best_pos = pos
    
    if best_pos is None:
        # Fallback to square grid
        m = int(math.ceil(math.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:n]
    
    return best_pos


def _local_optimize(pos, max_iter=1000, tol=1e-10):
    """
    Local optimization using a simple gradient-free method.
    We'll use a coordinate descent approach with small perturbations.
    """
    pos = pos.copy()
    current_score = _score(pos)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            for axis in range(2):
                # Try small moves in both directions
                for direction in [-1, 1]:
                    for step_size in [0.001, 0.0001, 0.00001]:
                        pos_trial = pos.copy()
                        pos_trial[i, axis] += direction * step_size
                        
                        # Keep in bounds
                        pos_trial[i, axis] = np.clip(pos_trial[i, axis], -0.5, 0.5)
                        
                        trial_score = _score(pos_trial)
                        if trial_score > current_score + tol:
                            pos = pos_trial
                            current_score = trial_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return pos


def construct():
    """Construct the circle centers."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Step 1: Generate initial configuration
    pos = _hexagonal_init(N)
    current_score = _score(pos)
    
    # Step 2: Run MD simulation to resolve overlaps
    # Use a small target radius for MD
    pos = _md_simulate(pos, dt=0.0005, steps=20000, damping=0.998)
    md_score = _score(pos)
    
    if md_score > current_score:
        pos = pos
        current_score = md_score
    
    # Step 3: Local optimization
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        pos = _local_optimize(pos, max_iter=500)
        local_score = _score(pos)
        
        if local_score > current_score:
            current_score = local_score
    
    # Step 4: If we have time, try multiple restarts with different initializations
    remaining_time = deadline - time.time()
    if remaining_time > 20.0:
        # Try a few different random perturbations
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the current best
            pos_perturbed = pos + rng.randn(N, 2) * 0.01
            pos_perturbed[:, 0] = np.clip(pos_perturbed[:, 0], -0.5 + 0.06, 0.5 - 0.06)
            pos_perturbed[:, 1] = np.clip(pos_perturbed[:, 1], -0.5 + 0.06, 0.5 - 0.06)
            
            # Quick MD
            pos_perturbed = _md_simulate(pos_perturbed, dt=0.0005, steps=5000, damping=0.998)
            pert_score = _score(pos_perturbed)
            
            if pert_score > current_score:
                pos = pos_perturbed
                current_score = pert_score
    
    return pos
