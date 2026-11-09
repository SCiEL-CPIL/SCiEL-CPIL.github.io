"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Physics-based circle packing: place circles with repulsive forces,
    then iteratively relax to maximize sum of radii while maintaining
    non-overlap and containment in unit square.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Initialize with a reasonable grid pattern
    side = int(np.ceil(np.sqrt(N)))
    positions = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                         for i in range(side) for j in range(side)][:N])
    
    # Start with small radii
    r = np.full(N, 0.05)
    
    best_sum = np.sum(r)
    best_positions = positions.copy()
    best_radii = r.copy()
    
    # Physics-based optimization
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)  # avoid division by zero
        
        # Repulsive forces: push apart overlapping circles
        overlap = np.maximum(r[:, np.newaxis] + r[np.newaxis, :] - dist, 0)
        
        # Gradient of objective: want to increase radii, but constrained by overlaps
        # Use a simple approach: if circles overlap, shrink them and move apart
        force = np.zeros_like(positions)
        radius_change = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] > 0:
                    # If overlapping, push apart and reduce radii
                    if dist[i, j] < r[i] + r[j]:
                        # Normalize direction
                        direction = diff[i, j] / dist[i, j]
                        # Push apart
                        force[i] += direction * 0.1
                        force[j] -= direction * 0.1
                        # Reduce radii proportionally to overlap
                        overlap_amt = r[i] + r[j] - dist[i, j]
                        radius_change[i] -= overlap_amt * 0.05
                        radius_change[j] -= overlap_amt * 0.05
        
        # Wall constraints: keep circles inside [0,1]^2
        for i in range(N):
            if positions[i, 0] < r[i]:
                force[i, 0] += 0.1 * (r[i] - positions[i, 0])
                radius_change[i] -= 0.02 * (r[i] - positions[i, 0])
            elif positions[i, 0] > 1 - r[i]:
                force[i, 0] -= 0.1 * (positions[i, 0] - (1 - r[i]))
                radius_change[i] -= 0.02 * (positions[i, 0] - (1 - r[i]))
            
            if positions[i, 1] < r[i]:
                force[i, 1] += 0.1 * (r[i] - positions[i, 1])
                radius_change[i] -= 0.02 * (r[i] - positions[i, 1])
            elif positions[i, 1] > 1 - r[i]:
                force[i, 1] -= 0.1 * (positions[i, 1] - (1 - r[i]))
                radius_change[i] -= 0.02 * (positions[i, 1] - (1 - r[i]))
        
        # Update positions and radii
        positions += force
        r += radius_change
        
        # Ensure radii are positive
        r = np.maximum(r, 0.01)
        
        # Ensure positions are within bounds (clamp)
        positions[:, 0] = np.clip(positions[:, 0], r, 1 - r)
        positions[:, 1] = np.clip(positions[:, 1], r, 1 - r)
        
        # Check if valid and update best
        if _is_valid(positions, r):
            current_sum = np.sum(r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_positions = positions.copy()
                best_radii = r.copy()
    
    # Final validation and cleanup
    if not _is_valid(best_positions, best_radii):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    result = np.column_stack([best_positions, best_radii])
    return result


def _is_valid(positions, radii):
    """Check if configuration is valid: no overlaps, all inside unit square."""
    n = len(positions)
    
    # Check bounds
    for i in range(n):
        if positions[i, 0] < radii[i] - 1e-6 or positions[i, 0] > 1 - radii[i] + 1e-6:
            return False
        if positions[i, 1] < radii[i] - 1e-6 or positions[i, 1] > 1 - radii[i] + 1e-6:
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                return False
    
    return True
# EVOLVE-BLOCK-END
