"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_radius(centres):
    """Compute the maximum radius for given centres."""
    # Clip to valid range
    centres = np.clip(centres, -0.5, 0.5)
    
    # Compute pairwise distances
    n = len(centres)
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Compute distance to boundary
    min_boundary = np.inf
    for i in range(n):
        for k in range(2):
            dist_to_wall = 0.5 - abs(centres[i][k])
            if dist_to_wall < min_boundary:
                min_boundary = dist_to_wall
    
    return min(min_pair_dist / 2.0, min_boundary)


def _initial_grid():
    """Create an initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_hexagonal():
    """Create an initial hexagonal packing arrangement."""
    # Hexagonal packing for 50 circles
    # Try different row configurations
    best_centres = None
    best_r = -1
    
    # Try to pack in a hexagonal pattern
    # Rows with alternating offsets
    for n_rows in range(5, 12):
        # Distribute circles among rows
        base_per_row = N // n_rows
        extra = N % n_rows
        row_counts = [base_per_row + (1 if i < extra else 0) for i in range(n_rows)]
        
        if any(c <= 0 for c in row_counts):
            continue
        
        centres = []
        row_height = 1.0 / (n_rows + 0.5)  # approximate
        for row_idx, count in enumerate(row_counts):
            y = -0.5 + (row_idx + 0.5) * (1.0 / n_rows)
            # Offset alternating rows
            offset = 0.0 if row_idx % 2 == 0 else 0.5 / count
            x_spacing = 1.0 / (count + 1)
            for i in range(count):
                x = -0.5 + (i + 1) * x_spacing + offset * x_spacing
                x = max(-0.5, min(0.5, x))
                y = max(-0.5, min(0.5, y))
                centres.append([x, y])
        
        if len(centres) == N:
            centres = np.array(centres)
            r = _compute_radius(centres)
            if r > best_r:
                best_r = r
                best_centres = centres
    
    if best_centres is None:
        return _initial_grid()
    return best_centres


def _repulsion_force(centres, r_target, wall_penalty=10.0):
    """Compute repulsion forces to push circles apart and away from walls."""
    n = len(centres)
    forces = np.zeros_like(centres)
    
    for i in range(n):
        for j in range(i+1, n):
            d = centres[i] - centres[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random small perturbation
                d = np.random.randn(2) * 1e-5
                dist = np.linalg.norm(d)
            
            # If too close, push apart
            if dist < 2 * r_target:
                force_mag = (2 * r_target - dist) / dist
                forces[i] += d * force_mag
                forces[j] -= d * force_mag
        
        # Wall repulsion
        for k in range(2):
            if centres[i][k] + r_target > 0.5:
                forces[i][k] -= (centres[i][k] + r_target - 0.5) * wall_penalty
            if centres[i][k] - r_target < -0.5:
                forces[i][k] += (-0.5 - (centres[i][k] - r_target)) * wall_penalty
    
    return forces


def _optimize_centres(centres, r_target, max_iters=5000, lr=0.01, deadline=None):
    """Optimize centres to achieve target radius."""
    n = len(centres)
    best_centres = centres.copy()
    best_score = _compute_radius(centres)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = _repulsion_force(centres, r_target)
        
        # Update positions
        centres = centres + lr * forces
        
        # Keep within bounds
        centres = np.clip(centres, -0.5, 0.5)
        
        # Check improvement
        current_score = _compute_radius(centres)
        if current_score > best_score:
            best_score = current_score
            best_centres = centres.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_centres, best_score


def _binary_search_radius(centres_init, deadline):
    """Binary search for the maximum achievable radius."""
    # Initial bounds
    r_low = 0.05
    r_high = 0.08
    
    # First, try to improve with current arrangement
    centres, r = _optimize_centres(centres_init, r_high, max_iters=2000, lr=0.01, deadline=deadline)
    
    # Binary search
    for _ in range(15):
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2
        centres_test, r_achieved = _optimize_centres(centres_init.copy(), r_mid, 
                                                      max_iters=1500, lr=0.01, deadline=deadline)
        
        if r_achieved >= r_mid - 1e-6:
            r_low = r_mid
            centres = centres_test
        else:
            r_high = r_mid
    
    return centres, _compute_radius(centres)


def construct():
    """Construct the packing of 50 circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Generate multiple initial configurations
    initial_configs = [
        _initial_grid(),
        _initial_hexagonal(),
    ]
    
    # Add some random perturbations
    for _ in range(3):
        if time.time() > deadline:
            break
        base = _initial_grid()
        noise = np.random.randn(N, 2) * 0.02
        perturbed = np.clip(base + noise, -0.5, 0.5)
        initial_configs.append(perturbed)
    
    best_centres = initial_configs[0]
    best_radius = _compute_radius(best_centres)
    
    # Try each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Optimize this configuration
        centres, radius = _optimize_centres(config.copy(), 0.075, 
                                             max_iters=3000, lr=0.01, deadline=deadline)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = centres
    
    # Final refinement with binary search
    if time.time() < deadline - 10:
        final_centres, final_radius = _binary_search_radius(best_centres.copy(), deadline)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centres = final_centres
    
    return best_centres
# EVOLVE-BLOCK-END
