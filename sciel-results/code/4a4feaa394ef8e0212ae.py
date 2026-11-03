import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with a grid
    # 6x5 grid = 30 spots, take 26
    # Or 5x5 grid + 1 random
    # Let's use 5x5 grid for 25, and 1 in the middle of one of the cells?
    # No, just 26 random circles is a valid start, but might be infeasible.
    # Grid is safer.
    
    # Create 5x5 grid
    side = 5
    grid_points = []
    for i in range(side):
        for j in range(side):
            grid_points.append([(i + 0.5) / side, (j + 0.5) / side])
    # Add 1 more point, e.g., at (0.5, 0.5) is occupied.
    # Let's just take the first 26 from a 6x5 grid.
    side_x, side_y = 6, 5
    points = []
    for i in range(side_x):
        for j in range(side_y):
            points.append([(i + 0.5) / side_x, (j + 0.5) / side_y])
            if len(points) == N:
                break
        if len(points) == N:
            break
            
    r_init = 0.5 / max(side_x, side_y) - 0.01
    circles = np.zeros((N, 3))
    for k in range(N):
        circles[k, 0] = points[k][0]
        circles[k, 1] = points[k][1]
        circles[k, 2] = r_init
        
    # Precompute indices for overlap
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def get_energy(c):
        radii = c[:, 2]
        pos = c[:, :2]
        
        e_sum_r = -np.sum(radii)
        
        # Boundary
        d_x = np.minimum(pos[:, 0], 1 - pos[:, 0])
        d_y = np.minimum(pos[:, 1], 1 - pos[:, 1])
        wall_viol = np.maximum(0, radii - d_x) + np.maximum(0, radii - d_y)
        e_boundary = 10.0 * np.sum(wall_viol**2)
        
        # Overlap
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        viol = np.maximum(0, r_sum[i_idx, j_idx] - dist[i_idx, j_idx])
        e_overlap = 10.0 * np.sum(viol**2)
        
        return e_sum_r + e_boundary + e_overlap, e_boundary, e_overlap

    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    T = 0.05
    cooling_rate = 0.99995
    pos_step = 0.05
    rad_step = 0.05
    
    current_energy, _, _ = get_energy(circles)
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Generate move
        idx = np.random.randint(N)
        new_circles = circles.copy()
        
        new_circles[idx, 0] += np.random.normal(0, pos_step)
        new_circles[idx, 1] += np.random.normal(0, pos_step)
        new_circles[idx, 2] += np.random.normal(0, rad_step)
        
        # Clip
        new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
        new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
        new_circles[:, 2] = np.clip(new_circles[:, 2], 0, 0.5)
        
        new_energy, e_b, e_o = get_energy(new_circles)
        dE = new_energy - current_energy
        
        # Accept
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            circles = new_circles
            current_energy = new_energy
            
            # Check if feasible
            if e_b < 1e-8 and e_o < 1e-8:
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
        
        # Cool
        T *= cooling_rate
        if T < 1e-6:
            T = 0.01 # Restart temperature occasionally? Or just break.
            # Restarting T might help escape local minima.
            # But we have a time limit. Let's just let it cool to 0.
            # Actually, if T becomes too small, we stop accepting bad moves.
            # We can periodically reheat.
            if step % 10000 == 0:
                T = 0.05

    return best_circles
# EVOLVE-BLOCK-END
