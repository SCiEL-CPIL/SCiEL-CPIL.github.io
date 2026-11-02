import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: Flower pattern
    positions = np.zeros((N, 2))
    rotations = np.zeros(N)
    
    # Center
    positions[0] = [0, 0]
    
    # Ring of 6
    for i in range(1, 7):
        angle = i * np.pi / 3
        positions[i] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
        
    # Add 5 more
    for i in range(7, N):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.sqrt(3) * 2
        positions[i] = [dist * np.cos(angle), dist * np.sin(angle)]
        
    outer_angle = 0.0
    
    def get_required_side(pos, rot, outer_rot):
        # Generate all vertices: (N, 6, 2)
        angles = rot[:, None] + np.arange(6)[None, :] * (np.pi / 3)
        vertices = pos[:, None, :] + np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        # vertices shape: (N, 6, 2) -> flatten to (N*6, 2)
        verts_flat = vertices.reshape(-1, 2)
        
        # Outer normals: (6, 2)
        n_angles = outer_rot + np.arange(6) * (np.pi / 3)
        normals = np.stack([np.cos(n_angles), np.sin(n_angles)], axis=-1)
        
        # Projections: (N*6, 6)
        projs = np.abs(verts_flat @ normals.T)
        
        max_proj = np.max(projs)
        return max_proj * 2.0 / np.sqrt(3.0)

    best_side = get_required_side(positions, rotations, outer_angle)
    best_pos = positions.copy()
    best_rot = rotations.copy()
    best_outer_rot = outer_angle
    
    step_size = 0.5
    
    while time.time() < deadline:
        new_pos = best_pos.copy()
        new_rot = best_rot.copy()
        new_outer_rot = best_outer_rot
        
        # Randomly pick a few hexagons to move
        for _ in range(2):
            idx = np.random.randint(0, N)
            new_pos[idx] += np.random.uniform(-step_size, step_size, 2)
            new_rot[idx] += np.random.uniform(-0.1, 0.1)
            
        new_outer_rot += np.random.uniform(-0.05, 0.05)
        
        new_side = get_required_side(new_pos, new_rot, new_outer_rot)
        
        if new_side < best_side:
            best_side = new_side
            best_pos = new_pos
            best_rot = new_rot
            best_outer_rot = new_outer_rot
            step_size = max(0.01, step_size * 0.995)
        else:
            # Small chance to restart or jump
            if np.random.random() < 0.005:
                # Random restart of a few points
                for _ in range(3):
                    idx = np.random.randint(0, N)
                    new_pos[idx] = np.random.uniform(-3, 3, 2)
                    new_rot[idx] = np.random.uniform(0, 2*np.pi)
                new_side = get_required_side(new_pos, new_rot, new_outer_rot)
                if new_side < best_side:
                    best_side = new_side
                    best_pos = new_pos
                    best_rot = new_rot
                    step_size = 0.5

    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], np.degrees(best_rot[i])])
    
    center = [0.0, 0.0]
    side = best_side
    angle = np.degrees(best_outer_rot)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
