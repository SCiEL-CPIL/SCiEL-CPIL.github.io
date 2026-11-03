import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    """Generate vertices of a unit hexagon centered at (x,y) with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit
    
    # Initial guess: 7-hex flower + 4 more
    d = np.sqrt(3)  # distance between adjacent hex centers in aligned grid
    
    # 7-hex cluster: center + 6 around
    init_centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5)
    ]
    
    # Add 4 more hexagons - place them to extend the hexagonal pattern
    # In a hexagonal grid, next ring has 12 hexagons. We add 4 strategic ones.
    additional_centers = [
        (2*d, 0),
        (3*d/2, 3.0),
        (-3*d/2, 3.0),
        (0, 4.5)
    ]
    
    all_centers = init_centers + additional_centers
    init_params = []
    for i in range(N):
        x, y = all_centers[i]
        ang = 0.0
        init_params.extend([x, y, ang])
    
    init_params = np.array(init_params)
    
    # Normals for outer hexagon (flat-top, centered at origin)
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def calc_side_and_penalty(params):
        """Calculate outer hexagon side length and overlap penalty."""
        pts = params.reshape(N, 3)
        xs = pts[:, 0]
        ys = pts[:, 1]
        thetas = pts[:, 2]
        
        # Calculate required outer hexagon side
        max_proj = 0.0
        for k in range(N):
            verts = hex_vertices(xs[k], ys[k], thetas[k])
            projs = verts @ normals.T  # 6x6
            max_proj = max(max_proj, np.max(projs))
        
        # Apothem A = max_proj, Side S = 2 * A / sqrt(3)
        side = 2.0 * max_proj / np.sqrt(3)
        
        # Calculate overlap penalty using center distance (conservative)
        # Minimum center distance for non-overlapping unit hexagons is sqrt(3)
        # when aligned; with rotation it can be less, but sqrt(3) is a safe lower bound
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        return side, penalty
    
    def objective(params):
        side, penalty = calc_side_and_penalty(params)
        return side + 50.0 * penalty
    
    best_side = float('inf')
    best_params = init_params.copy()
    
    np.random.seed(42)
    
    # Multiple restart strategies
    restart_configs = []
    
    # Strategy 0: Original hexagonal grid
    restart_configs.append(init_params.copy())
    
    # Strategy 1: Slightly perturbed
    restart_configs.append(init_params + np.random.normal(0, 0.3, N*3))
    restart_configs[-1][2::3] = 0.0
    
    # Strategy 2: Rotated 30 degrees
    angle = np.deg2rad(30)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    rotated = []
    for i in range(N):
        x, y, a = init_params[i*3], init_params[i*3+1], init_params[i*3+2]
        new_x, new_y = R @ np.array([x, y])
        new_a = a + 30
        rotated.extend([new_x, new_y, new_a])
    restart_configs.append(np.array(rotated))
    
    # Strategy 3: More compact initial guess
    compact = init_params.copy()
    # Move outer hexagons slightly inward
    for i in range(1, N):
        x, y = compact[i*3], compact[i*3+1]
        dist = np.sqrt(x*x + y*y)
        if dist > 0:
            scale = 0.95
            compact[i*3] = x * scale
            compact[i*3+1] = y * scale
    restart_configs.append(compact)
    
    # Strategy 4: Different arrangement - 5 in a row, 3 above, 3 below
    positions4 = [
        (0, 0),
        (-d, 0), (d, 0),
        (-d/2, 1.5), (d/2, 1.5),
        (0, 1.5),
        (-d/2, -1.5), (d/2, -1.5),
        (0, -1.5),
        (-2*d, 0), (2*d, 0)
    ]
    params4 = np.zeros(N*3)
    for i in range(N):
        params4[i*3] = positions4[i][0]
        params4[i*3+1] = positions4[i][1]
        params4[i*3+2] = 0
    restart_configs.append(params4)
    
    # Strategy 5: Random scatter
    params5 = np.zeros(N*3)
    params5[0] = 0
    params5[1] = 0
    for i in range(1, N):
        ang = 2*np.pi*i/(N-1)
        rad = 2.0 + np.random.uniform(-0.3, 0.3)
        params5[i*3] = rad * np.cos(ang)
        params5[i*3+1] = rad * np.sin(ang)
        params5[i*3+2] = np.random.uniform(-15, 15)
    restart_configs.append(params5)
    
    # Strategy 6: Another hexagonal pattern
    positions6 = [
        (0, 0),
        (d, 0), (-d, 0),
        (d/2, 1.5), (-d/2, 1.5),
        (d/2, -1.5), (-d/2, -1.5),
        (2*d, 0), (-2*d, 0),
        (0, 3.0), (0, -3.0)
    ]
    params6 = np.zeros(N*3)
    for i in range(N):
        params6[i*3] = positions6[i][0]
        params6[i*3+1] = positions6[i][1]
        params6[i*3+2] = 0
    restart_configs.append(params6)
    
    # Run optimization for each restart
    for r, current_params in enumerate(restart_configs):
        if time.time() > deadline - 10:
            break
        
        try:
            # First pass: Nelder-Mead
            res = scipy.optimize.minimize(
                objective,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            side, pen = calc_side_and_penalty(res.x)
            if pen < 1e-3 and side < best_side:
                best_side = side
                best_params = res.x.copy()
            
            # Second pass: Polish with tighter tolerance if we have time
            if time.time() < deadline - 15 and pen < 1e-3:
                res2 = scipy.optimize.minimize(
                    objective,
                    res.x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7}
                )
                side2, pen2 = calc_side_and_penalty(res2.x)
                if pen2 < 1e-3 and side2 < best_side:
                    best_side = side2
                    best_params = res2.x.copy()
                    
        except Exception:
            pass
    
    # Additional random restarts if we have time
    while time.time() < deadline - 15:
        if best_side < float('inf'):
            current_params = best_params + np.random.normal(0, 0.5, N*3)
            current_params[2::3] = np.random.uniform(-15, 15, N)
        else:
            current_params = init_params + np.random.normal(0, 1.0, N*3)
            current_params[2::3] = np.random.uniform(-15, 15, N)
        
        try:
            res = scipy.optimize.minimize(
                objective,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            side, pen = calc_side_and_penalty(res.x)
            if pen < 1e-3 and side < best_side:
                best_side = side
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish if we found a valid solution
    if best_side < float('inf') and time.time() < deadline - 5:
        try:
            res_final = scipy.optimize.minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            side_final, pen_final = calc_side_and_penalty(res_final.x)
            if pen_final < 1e-3 and side_final < best_side:
                best_side = side_final
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Construct output
    if best_side == float('inf'):
        # Fallback: return a valid (but not optimal) packing
        # Use the initial hexagonal grid
        inner = [[all_centers[i][0], all_centers[i][1], 0.0] for i in range(N)]
        # Calculate the actual side needed
        max_proj = 0.0
        for i in range(N):
            x, y, ang = inner[i]
            verts = hex_vertices(x, y, ang)
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
        side = 2.0 * max_proj / np.sqrt(3)
        return inner, (0.0, 0.0), float(side), 0.0
    
    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])
        
    return inner, (0.0, 0.0), float(best_side), 0.0
