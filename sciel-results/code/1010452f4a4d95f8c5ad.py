"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using a hexagonal lattice.
2. Build a static neighbor list to reduce O(N^2) computations to O(N*k).
3. Refine with a vectorized soft-min gradient ascent (log-sum-exp approximation).
4. Use a deadline guard to ensure termination within 120s.
"""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin before 120s limit (150s kill has 30s buffer, but 120s is the stated limit)

    # --- Step 1: Generate good initial positions ---
    # Use a hexagonal lattice.
    
    def hexagonal_positions(n_target):
        """Generate n_target points in a hexagonal pattern within the unit disk."""
        # Estimate spacing based on area
        base_spacing = 1.0 / np.sqrt(n_target * 2.0 / np.pi)
        
        best_points = None
        best_count_diff = float('inf')
        
        # Try a range of spacings around the theoretical base
        for spacing_mult in np.linspace(0.85, 1.15, 20):
            spacing = base_spacing * spacing_mult
            
            # Generate hex lattice points
            a = spacing
            b = a * np.sqrt(3) / 2
            k = int(1.0 / spacing) + 5
            points = []
            for j in range(-k, k + 1):
                y = j * b
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    if x*x + y*y <= 0.98**2:  # Keep well inside for now
                        points.append((x, y))
            
            points = np.array(points)
            if len(points) == 0:
                continue
            
            # Sort by distance from center
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            points = points[idx]
            
            if len(points) >= n_target:
                points = points[:n_target]
                diff = abs(len(points) - n_target)
                if diff < best_count_diff:
                    best_count_diff = diff
                    best_points = points
            
            # If we have exactly n_target or very close, use it
            if len(points) == n_target:
                best_points = points
                break
        
        if best_points is None or len(best_points) != n_target:
            # Fallback: create a simple radial pattern
            angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
            radii = np.linspace(0.05, 0.95, n_target)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        return best_points
    
    positions = hexagonal_positions(N)
    
    # --- Step 2: Compute initial radius and setup ---
    
    def compute_radius(pos):
        """Compute the current minimum radius."""
        # Distance to boundary
        boundary = 1.0 - np.linalg.norm(pos, axis=1)
        min_boundary = np.min(boundary)
        
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Set diagonal to infinity
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist) / 2.0
        
        return min(min_boundary, min_pair_dist)
    
    current_r = compute_radius(positions)
    if current_r < 0.04:
        # Scale positions to ensure we have a valid starting point
        max_norm = np.max(np.linalg.norm(positions, axis=1))
        if max_norm > 0:
            positions = positions * (0.95 / max_norm)
        current_r = compute_radius(positions)
    
    best_positions = positions.copy()
    best_r = current_r
    
    # --- Step 3: Build Neighbor List ---
    # We assume the local topology (which points are neighbors) remains roughly stable.
    # We define neighbors as points within a certain radius in the initial configuration.
    # This avoids O(N^2) checks every step.
    # Let's use a radius slightly larger than the initial spacing to be safe.
    
    # Estimate initial spacing from the hex lattice or just use a heuristic
    # For N=300, spacing is roughly 1/sqrt(300) ~ 0.058.
    # Let's compute actual min distance to set neighbor radius.
    diff_init = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_init = np.linalg.norm(diff_init, axis=2)
    np.fill_diagonal(dists_init, np.inf)
    min_dist_init = np.min(dists_init)
    
    # Use a factor of 3.0 to ensure we capture all relevant interactions
    neighbor_radius = min_dist_init * 3.0
    
    i_idx, j_idx = np.where(dists_init < neighbor_radius)
    # Ensure unique pairs i < j
    mask = i_idx < j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    # If for some reason no neighbors were found (shouldn't happen), fallback to full pairs
    if len(i_idx) == 0:
        i_idx, j_idx = np.triu_indices(N, k=1)

    # --- Step 4: Optimization Loop ---
    # Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    # We use a soft-min approximation to make the objective smooth and differentiable.
    # soft_min(x) = -1/k * log(sum(exp(-k * x)))
    # Gradient of soft_min w.r.t x_i is proportional to exp(-k * x_i) / sum(exp(-k * x_j))
    
    alpha = 0.05  # Initial learning rate
    cooling = 0.9995
    k_soft = 50.0  # Start with moderate sharpness
    k_polish = 200.0 # Sharper for final phase
    steps_per_k = 1000 # Increase sharpness every 1000 steps
    
    step_count = 0
    patience = 0
    max_patience = 5000
    
    while time.time() - start_time < deadline:
        # Check if we should increase sharpness
        if step_count > 0 and step_count % steps_per_k == 0:
            k_soft = min(k_soft * 1.2, k_polish)
        
        # Compute distances for neighbor pairs only
        # diff[i, j] = p_i - p_j
        diff = positions[i_idx] - positions[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        safe_dists = np.maximum(dists, 1e-9)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        
        # Boundary distances
        rad = np.linalg.norm(positions, axis=1)
        b_vals = 1.0 - rad
        
        # Combine pair distances (d/2) and boundary distances (b) into a single list of values
        # Note: The objective is min(min(d_ij)/2, min(b_i))
        # So we are taking the min of the set {d_ij/2 for all i<j} U {b_i for all i}
        
        # For efficiency, we compute weights for the soft min over these combined values.
        # However, the gradient contributions are different.
        # Let's compute the soft min value and weights for the combined set.
        
        # Values:
        # pair_vals = dists / 2.0
        # bound_vals = b_vals
        
        pair_vals = dists / 2.0
        all_vals = np.concatenate([pair_vals, b_vals])
        
        # Compute weights: w_i = exp(-k * v_i) / sum(exp(-k * v_j))
        # To avoid overflow/underflow, subtract the max (which is the min value * -k)
        # Actually, exp(-k * v) is largest for smallest v.
        # So we can just compute exp(-k * (v - v_min))
        v_min = np.min(all_vals)
        exponents = -k_soft * (all_vals - v_min)
        weights = np.exp(exponents)
        sum_weights = np.sum(weights)
        
        if sum_weights < 1e-9:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= sum_weights
            
        # Split weights back
        pair_weights = weights[:len(dists)]
        bound_weights = weights[len(dists):]
        
        # Compute gradients
        # grad_{p_i} (d_ij/2) = 0.5 * (p_i - p_j) / d_ij
        # grad_{p_i} (b_i) = -p_i / ||p_i||
        
        grads = np.zeros_like(positions)
        
        # Pairwise gradients
        # Contribution to p_i: +0.5 * unit_vec * weight
        # Contribution to p_j: -0.5 * unit_vec * weight
        half_unit_vecs = 0.5 * unit_vecs
        np.add.at(grads, i_idx, half_unit_vecs * pair_weights[:, np.newaxis])
        np.add.at(grads, j_idx, -half_unit_vecs * pair_weights[:, np.newaxis])
        
        # Boundary gradients
        safe_rad = np.maximum(rad, 1e-9)
        bound_vecs = -positions / safe_rad[:, np.newaxis]
        grads += bound_vecs * bound_weights[:, np.newaxis]
        
        # Update positions
        positions += alpha * grads
        
        # Project back inside circle (hard constraint)
        rad_new = np.linalg.norm(positions, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            positions[mask] /= rad_new[mask, np.newaxis]
        
        # Evaluate current radius
        current_r = compute_radius(positions)
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_positions = positions.copy()
            patience = 0
        else:
            patience += 1
            
        if patience > max_patience:
            break
            
        # Cooling
        alpha *= cooling
        
        step_count += 1
        
        # Periodic rebuild of neighbor list?
        # It's expensive, but might help if topology changes significantly.
        # Let's try rebuilding every 10,000 steps if time permits.
        if step_count % 10000 == 0 and time.time() - start_time < deadline - 10:
             diff_curr = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
             dists_curr = np.linalg.norm(diff_curr, axis=2)
             np.fill_diagonal(dists_curr, np.inf)
             min_dist_curr = np.min(dists_curr)
             neighbor_radius = min_dist_curr * 3.0
             i_idx, j_idx = np.where(dists_curr < neighbor_radius)
             mask = i_idx < j_idx
             i_idx = i_idx[mask]
             j_idx = j_idx[mask]
             if len(i_idx) == 0:
                 i_idx, j_idx = np.triu_indices(N, k=1)

    # Final check
    final_r = compute_radius(best_positions)
    if final_r > compute_radius(positions):
        return best_positions
    else:
        return positions

# Ensure the function returns a valid array
def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        # Fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Check for non-finite values
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # Check inside unit disk
    norms = np.linalg.norm(pos, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
        
    # Check for coincident points
    # If any pair is too close, jitter them
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-4:
        # Find indices of close pairs and jitter
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-4:
                    pos[i] += np.random.randn(2) * 1e-3
                    pos[j] -= np.random.randn(2) * 1e-3
                    # Re-check boundary
                    n = np.linalg.norm(pos[i])
                    if n > 0.999:
                        pos[i] = pos[i] * (0.999 / n)
                    n = np.linalg.norm(pos[j])
                    if n > 0.999:
                        pos[j] = pos[j] * (0.999 / n)
                        
    return pos

# Wrap the construct function to ensure validity
_original_construct = construct

def construct():
    try:
        pos = _original_construct()
        pos = _validate_and_fix(pos)
        return pos
    except Exception:
        # Fallback to a simple valid configuration
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.05, 0.95, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pos
# EVOLVE-BLOCK-END
