import numpy as np

def construct():
    N = 26
    
    # Hexagonal initialization
    # We want to pack 26 circles.
    # Let's try to fit them in a hex grid.
    # 4 rows of 5? 20.
    # 5 rows of 5? 25.
    # 5 rows of 5 + 1?
    
    # Let's generate a hex grid with spacing d.
    # We want the grid to fit in [0,1]^2.
    # Let's try d = 0.2.
    # Row height = d * sqrt(3)/2 ≈ 0.1732.
    # 5 rows: height = 4 * 0.1732 + d ≈ 0.69 + 0.2 = 0.89. Fits.
    # 5 cols: width = 4 * d + d = 5d = 1.0. Fits.
    # So a 5x5 hex grid fits.
    # We have 25 circles. We need 26.
    # The 26th circle can be placed in the center of the square, but it will overlap.
    # The "Grow and Repel" will push it out.
    
    # Let's use a 5x5 hex grid for the first 25, and place the 26th at (0.5, 0.5) with a small radius?
    # Or just use a 6x5 grid and drop 4.
    
    # Let's use a 6x5 grid (30 circles) and remove 4 corners.
    # This gives a good distribution.
    
    # Generate 6x5 grid
    cols = 6
    rows = 5
    x_coords = np.linspace(0.1, 0.9, cols)
    y_coords = np.linspace(0.1, 0.9, rows)
    
    points = []
    for y in y_coords:
        for x in x_coords:
            points.append((x, y))
            
    # points has 30 items.
    # Remove 4 corners: (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)
    # Indices:
    # (0.1, 0.1) is index 0
    # (0.9, 0.1) is index 5
    # (0.1, 0.9) is index 25
    # (0.9, 0.9) is index 29
    
    # Let's just filter them out.
    filtered_points = []
    for p in points:
        # Check if it's a corner
        is_corner = False
        if abs(p[0] - 0.1) < 1e-5 and abs(p[1] - 0.1) < 1e-5: is_corner = True
        if abs(p[0] - 0.9) < 1e-5 and abs(p[1] - 0.1) < 1e-5: is_corner = True
        if abs(p[0] - 0.1) < 1e-5 and abs(p[1] - 0.9) < 1e-5: is_corner = True
        if abs(p[0] - 0.9) < 1e-5 and abs(p[1] - 0.9) < 1e-5: is_corner = True
        
        if not is_corner:
            filtered_points.append(p)
            
    # We should have 26 points now.
    if len(filtered_points) != N:
        # Fallback to random
        np.random.seed(42)
        filtered_points = [tuple(p) for p in (np.random.rand(N, 2) * 0.8 + 0.1)]
        
    pos = np.array(filtered_points[:N])
    
    # Initial radii
    # Estimate max radius from grid spacing
    # Min distance in grid is 0.2. So r=0.1 is safe.
    r0 = 0.08
    radii = np.full(N, r0)
    
    # Grow and Repel
    max_iter = 2000
    step_size = 0.002
    damping = 0.5
    
    for it in range(max_iter):
        # Try to grow radii
        new_radii = radii + step_size
        radii = new_radii
        
        # Relaxation steps
        for _ in range(20):
            forces = np.zeros((N, 2))
            
            # Wall repulsion
            for i in range(N):
                if pos[i, 0] < radii[i]:
                    forces[i, 0] += (radii[i] - pos[i, 0])
                elif pos[i, 0] > 1 - radii[i]:
                    forces[i, 0] -= (pos[i, 0] - (1 - radii[i]))
                
                if pos[i, 1] < radii[i]:
                    forces[i, 1] += (radii[i] - pos[i, 1])
                elif pos[i, 1] > 1 - radii[i]:
                    forces[i, 1] -= (pos[i, 1] - (1 - radii[i]))
            
            # Circle repulsion
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = radii[i] + radii[j]
                    
                    if dist_sq < min_dist*min_dist and dist_sq > 1e-12:
                        dist = np.sqrt(dist_sq)
                        overlap = min_dist - dist
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                        
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    elif dist_sq <= 1e-12:
                        angle = np.random.rand() * 2 * np.pi
                        forces[i, 0] += np.cos(angle)
                        forces[i, 1] += np.sin(angle)
                        forces[j, 0] -= np.cos(angle)
                        forces[j, 1] -= np.sin(angle)
            
            pos += damping * forces * step_size
            pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
            pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)
            
        # Check for overlaps
        overlaps = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist*min_dist - 1e-6:
                    overlaps += 1
            
            if pos[i, 0] < radii[i] - 1e-6 or pos[i, 0] > 1 - radii[i] + 1e-6:
                overlaps += 1
            if pos[i, 1] < radii[i] - 1e-6 or pos[i, 1] > 1 - radii[i] + 1e-6:
                overlaps += 1
        
        if overlaps > 0:
            radii -= step_size
            step_size *= 0.95
            if step_size < 1e-5:
                break
        else:
            # Optional: increase step size slightly if stable?
            pass

    # Final cleanup
    for _ in range(50):
        forces = np.zeros((N, 2))
        for i in range(N):
            if pos[i, 0] < radii[i]: forces[i, 0] += (radii[i] - pos[i, 0])
            elif pos[i, 0] > 1 - radii[i]: forces[i, 0] -= (pos[i, 0] - (1 - radii[i]))
            if pos[i, 1] < radii[i]: forces[i, 1] += (radii[i] - pos[i, 1])
            elif pos[i, 1] > 1 - radii[i]: forces[i, 1] -= (pos[i, 1] - (1 - radii[i]))
            
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist*min_dist and dist_sq > 1e-12:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        pos += 0.5 * forces * 0.01
        pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
        pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)
        
    result = np.column_stack((pos[:, 0], pos[:, 1], radii))
    return result
