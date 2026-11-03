"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate a robust hexagonal lattice initialization.
2. Refine with a vectorized soft-min gradient ascent using a dynamic neighbor list.
3. Use a deadline guard to ensure termination within 150s.
"""
import numpy as np
import time
import sys

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill

    # --- Step 1: Generate good initial positions ---
    def hexagonal_positions(n_target, spacing_mult=1.0):
        """Generate n_target points in a hexagonal pattern within the unit disk."""
        # Estimate spacing based on area
        # Area of disk = pi * r^2 = pi.
        # Area of hex cell ~ sqrt(3)/2 * s^2.
        # n * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi / (n*sqrt(3)))
        base_spacing = np.sqrt(2.0 * np.pi / (n_target * np.sqrt(3.0)))
        spacing = base_spacing * spacing_mult
        
        a = spacing
        b = a * np.sqrt(3) / 2
        k = int(1.0 / spacing) + 5
        points = []
        for j in range(-k, k + 1):
            y = j * b
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * a
                if x*x + y*y <= 0.99**2:  # Keep well inside for now
                    points.append((x, y))
        
        points = np.array(points)
        if len(points) == 0:
            return None
        
        # Sort by distance from center to prefer central points for better packing
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        points = points[idx]
        
        if len(points) >= n_target:
            points = points[:n_target]
            return points
        
        return None

    # Try a few spacing multipliers to find a good start
    inits = []
    for mult in [0.95, 1.0, 1.05]:
        pts = hexagonal_positions(N, mult)
        if pts is not None and len(pts) == N:
            inits.append(pts)

    # Fallback
    if not inits:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.05, 0.95, N)
        inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

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
    
    best_positions = None
    best_r = 0.0

    # --- Optimization function for a single initialization ---
    def optimize_single(positions, time_limit):
        """Run gradient ascent on a single initialization until time_limit."""
        current_r = compute_radius(positions)
        if current_r < 0.03:
            # Scale positions to ensure we have a valid starting point
            max_norm = np.max(np.linalg.norm(positions, axis=1))
            if max_norm > 0:
                positions = positions * (0.95 / max_norm)
            current_r = compute_radius(positions)
        
        best_pos = positions.copy()
        best_r_local = current_r
        
        # --- Build Neighbor List ---
        def build_neighbors(pos):
            diff_init = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists_init = np.linalg.norm(diff_init, axis=2)
            np.fill_diagonal(dists_init, np.inf)
            min_dist_init = np.min(dists_init)
            neighbor_radius = min_dist_init * 2.5
            i_idx, j_idx = np.where(dists_init < neighbor_radius)
            mask = i_idx < j_idx
            i_idx = i_idx[mask]
            j_idx = j_idx[mask]
            if len(i_idx) == 0:
                i_idx, j_idx = np.triu_indices(N, k=1)
            return i_idx, j_idx

        i_idx, j_idx = build_neighbors(positions)

        # --- Optimization Loop ---
        alpha = 0.05  # Initial learning rate
        cooling = 0.9995
        k_soft = 50.0  # Start with moderate sharpness
        k_polish = 300.0 # Sharper for final phase
        steps_per_k = 500 # Increase sharpness every 500 steps
        
        step_count = 0
        patience = 0
        max_patience = 3000
        
        while time.time() - start_time < time_limit:
            # Check if we should increase sharpness
            if step_count > 0 and step_count % steps_per_k == 0:
                k_soft = min(k_soft * 1.15, k_polish)
            
            # Compute distances for neighbor pairs only
            diff = positions[i_idx] - positions[j_idx]
            dists = np.linalg.norm(diff, axis=1)
            
            # Avoid division by zero
            safe_dists = np.maximum(dists, 1e-9)
            unit_vecs = diff / safe_dists[:, np.newaxis]
            
            # Boundary distances
            rad = np.linalg.norm(positions, axis=1)
            b_vals = 1.0 - rad
            
            # Combine pair distances (d/2) and boundary distances (b)
            pair_vals = dists / 2.0
            all_vals = np.concatenate([pair_vals, b_vals])
            
            # Compute weights for soft min
            v_min = np.min(all_vals)
            exponents = -k_soft * (all_vals - v_min)
            weights = np.exp(exponents)
            sum_weights = np.sum(weights)
            
            if sum_weights < 1e-9:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights /= sum_weights
                
            pair_weights = weights[:len(dists)]
            bound_weights = weights[len(dists):]
            
            # Compute gradients
            grads = np.zeros_like(positions)
            
            # Pairwise gradients
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
            if current_r > best_r_local:
                best_r_local = current_r
                best_pos = positions.copy()
                patience = 0
            else:
                patience += 1
                
            if patience > max_patience:
                break
                
            # Cooling
            alpha *= cooling
            
            step_count += 1
            
            # Periodic rebuild of neighbor list
            if step_count % 2000 == 0 and time.time() - start_time < time_limit - 5:
                 i_idx, j_idx = build_neighbors(positions)

        return best_pos, best_r_local

    # --- Run optimization ---
    # We only have one good init usually, but let's run it with the full budget.
    # If we had multiple, we'd split time.
    init_pos = inits[0]
    pos, r = optimize_single(init_pos.copy(), deadline - 2.0)
    
    if r > best_r:
        best_r = r
        best_positions = pos

    # If no optimization ran or all failed, use the best initialization
    if best_positions is None:
        best_positions = inits[0]
        best_r = compute_radius(best_positions)

    # Final check
    final_r = compute_radius(best_positions)
    if final_r > best_r:
        best_r = final_r

    return best_positions

# Ensure the function returns a valid array
def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
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
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-4:
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-4:
                    pos[i] += np.random.randn(2) * 1e-3
                    pos[j] -= np.random.randn(2) * 1e-3
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
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.05, 0.95, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pos
# EVOLVE-BLOCK-END
