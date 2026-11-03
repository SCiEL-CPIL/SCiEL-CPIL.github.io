"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Initialize with Hexagonal Lattice
    # Hexagonal packing is denser than square grid.
    # Side length of unit square is 1.
    # We want to fit N points.
    # Hex lattice spacing d. Area per point ~ d^2 * sqrt(3)/2.
    # N * d^2 * sqrt(3)/2 <= 1 => d <= sqrt(2 / (N * sqrt(3)))
    # For N=200, d ~ sqrt(2 / (200 * 1.732)) ~ sqrt(0.00577) ~ 0.076
    # Radius r = d/2 ~ 0.038. This is a good upper bound estimate.
    
    # Generate hex lattice points
    d = 0.075 # Initial spacing estimate
    points = []
    # Rows in hex lattice are offset by d/2 in x, and separated by d*sqrt(3)/2 in y
    dy = d * np.sqrt(3) / 2
    max_y = 0.5
    min_y = -0.5
    max_x = 0.5
    min_x = -0.5
    
    row = 0
    y = min_y + dy/2
    while y < max_y and len(points) < N:
        offset = 0 if row % 2 == 0 else d/2
        x = min_x + offset
        while x < max_x and len(points) < N:
            # Check if inside square
            if min_x <= x <= max_x and min_y <= y <= max_y:
                points.append([x, y])
            x += d
        row += 1
        y += dy
        
    points = np.array(points)
    
    # If we didn't get enough points (unlikely with d=0.075 for N=200 in 1x1 square), 
    # or if we have too many (we capped at N), we need to handle it.
    # Actually, the loop above adds points until N is reached or space runs out.
    # If len(points) < N, we need to add random points or shrink d.
    # Let's ensure we have exactly N points.
    if len(points) < N:
        # Add random points in the square
        random_pts = np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))
        points = np.vstack([points, random_pts])
    elif len(points) > N:
        points = points[:N]

    # 2. Force-Directed Relaxation
    # Objective: Maximize min distance.
    # Strategy: Iteratively push apart points that are too close.
    # Use spatial hashing for efficiency.
    
    def get_neighbors(points, cell_size):
        # Simple spatial hash
        hash_map = {}
        for i, p in enumerate(points):
            cx = int(np.floor(p[0] / cell_size))
            cy = int(np.floor(p[1] / cell_size))
            key = (cx, cy)
            if key not in hash_map:
                hash_map[key] = []
            hash_map[key].append(i)
        return hash_map

    def compute_forces(points, target_dist):
        # target_dist is the current best min distance * 1.05 or similar
        # We want to increase the minimum distance.
        # Force is repulsive if dist < target_dist, else 0.
        # Actually, a better approach for packing is to maximize the minimum distance.
        # We can use a gradient ascent on the min distance, but it's non-differentiable.
        # Alternative: Minimize sum of 1/d^2 for pairs with d < r_target.
        # Or simply: For each pair with d < r_target, push them apart.
        
        forces = np.zeros_like(points)
        cell_size = target_dist
        hash_map = get_neighbors(points, cell_size)
        
        for i in range(len(points)):
            p = points[i]
            cx = int(np.floor(p[0] / cell_size))
            cy = int(np.floor(p[1] / cell_size))
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in hash_map:
                        for j in hash_map[key]:
                            if i == j:
                                continue
                            q = points[j]
                            diff = p - q
                            dist = np.linalg.norm(diff)
                            if dist < target_dist and dist > 1e-10:
                                # Repulsive force
                                # F = (target_dist - dist) * diff / dist
                                # This pushes them apart to reach target_dist
                                f = (target_dist - dist) * diff / dist
                                forces[i] += f
                                # Newton's third law
                                # We will accumulate forces for all i, so we don't need to add to j here if we iterate all i
                                # But to be symmetric and efficient, we can just compute for all i.
        return forces

    # Initial target distance: estimate from area
    # r ~ sqrt(1/(N*2)) * 1.1 (hex factor)
    r_init = np.sqrt(1.0 / N) * 0.8
    target_dist = 2 * r_init
    
    # Adaptive target distance
    # We want to increase target_dist gradually.
    # Start with a small target, relax, then increase.
    
    current_best_r = 0.0
    for step in range(500):
        if time.time() > deadline:
            break
            
        # Compute current min distance to update target
        # This is O(N^2) if naive, but we can use the hash grid or just do it every few steps
        if step % 10 == 0:
            # O(N^2) min distance check
            min_d = float('inf')
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    d = np.linalg.norm(points[i] - points[j])
                    if d < min_d:
                        min_d = d
            # Also check boundaries
            for p in points:
                d_b = min(p[0] + 0.5, 0.5 - p[0], p[1] + 0.5, 0.5 - p[1])
                if d_b < min_d:
                    min_d = d_b
            current_best_r = min_d / 2.0
            # Increase target distance slightly
            target_dist = min_d * 1.02
            
        # Compute forces
        forces = compute_forces(points, target_dist)
        
        # Boundary forces: Push points inside if they are close to boundary
        # If p[0] > 0.5 - r, push left. If p[0] < -0.5 + r, push right.
        # We use current_best_r as estimate for boundary constraint
        r_est = max(current_best_r, 1e-5)
        for i in range(len(points)):
            p = points[i]
            if p[0] > 0.5 - r_est:
                forces[i][0] -= (p[0] - (0.5 - r_est))
            elif p[0] < -0.5 + r_est:
                forces[i][0] += ((-0.5 + r_est) - p[0])
            if p[1] > 0.5 - r_est:
                forces[i][1] -= (p[1] - (0.5 - r_est))
            elif p[1] < -0.5 + r_est:
                forces[i][1] += ((-0.5 + r_est) - p[1])
                
        # Update positions
        # Learning rate should be small to avoid oscillation
        lr = 0.01
        points += lr * forces
        
        # Clip to square (hard constraint)
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)

    # Final check and return
    # Ensure valid shape
    return points

# EVOLVE-BLOCK-END
