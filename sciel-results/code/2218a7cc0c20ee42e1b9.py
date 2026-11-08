import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        return np.array([x, y])
    
    # Initial positions (axial)
    # A "hexagon" of side 2 has 19. We need 11.
    # Let's take the 11 most central ones.
    # Distance from center in axial coords: |q|+|r|+|q+r|
    # Generate all axial coords in a box and pick 11 closest to origin.
    
    coords = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            dist = abs(q) + abs(r) + abs(q+r)
            coords.append((dist, q, r))
    coords.sort()
    
    initial_axial = [c[1:] for c in coords[:N]]
    initial_positions = np.array([axial_to_cartesian(q, r) for q, r in initial_axial])
    
    # State: positions (N x 2)
    positions = initial_positions.copy()
    
    # Pre-compute normals for the 6 sides of the outer hexagon
    # For a regular hexagon with vertices at 0, 60, ..., the normals to the sides are at 30, 90, ...
    thetas = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(t), np.sin(t)] for t in thetas]) # Shape (6, 2)
    
    def get_side_length(positions, outer_center):
        # Transform positions to outer hexagon frame
        local_pos = positions - outer_center
        
        # Project onto 6 axes
        # projs shape: (N, 6)
        projs = local_pos @ normals.T
        
        # The required apothem is the max absolute projection over all points and all normals
        apothem = np.max(np.abs(projs))
        
        # Side length S = 2 * apothem / sqrt(3)
        side = 2 * apothem / np.sqrt(3)
        return side

    current_side = get_side_length(positions, np.mean(positions, axis=0))
    best_side = current_side
    best_positions = positions.copy()
    best_center = np.mean(positions, axis=0).copy()
    
    step_size = 0.5
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            # Try moving particle i
            # We can vectorize the candidate generation slightly, but N is small.
            # Let's just iterate over 4 directions.
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    # Create a copy of positions with particle i moved
                    # To avoid full copy, we can modify a view, but NumPy doesn't allow easy in-place modification of a slice of a 2D array without copying if we want to keep the original.
                    # However, N=11 is small. Copy is fast.
                    new_pos = positions.copy()
                    new_pos[i] += np.array([dx, dy])
                    
                    new_center = np.mean(new_pos, axis=0)
                    new_side = get_side_length(new_pos, new_center)
                    
                    if new_side < current_side:
                        positions = new_pos
                        current_side = new_side
                        improved = True
                        
                        if current_side < best_side:
                            best_side = current_side
                            best_positions = positions.copy()
                            best_center = new_center.copy()
                        
                        # If we improved, we might want to restart the inner loop or just continue.
                        # Continuing is fine, but we might miss other directions from the old position.
                        # For speed, we just continue.
        
        if not improved:
            step_size *= 0.8
            if step_size < 0.01:
                break
    
    # Return best result
    # Convert to required format
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], 0.0])
    
    return inner, (best_center[0], best_center[1]), best_side, 0.0
