"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 26
DEADLINE = time.time() + 110.0  # Safety margin

# EVOLVE-BLOCK-START

def get_hex_grid(n_points_target=40):
    """Generate a hexagonal grid of points within [0,1]^2."""
    points = []
    # Hexagonal packing spacing
    # Distance between centers in hex grid is d.
    # Rows are offset by d/2 in x and d*sqrt(3)/2 in y.
    # Let's try a range of spacings to find a good density.
    # For 26 circles in unit square, average area per circle is 1/26 ~ 0.038.
    # Circle area pi*r^2. If r ~ 0.1, area ~ 0.03.
    # So d ~ 2*r ~ 0.2.
    
    best_points = []
    
    # Try different grid densities
    for d in [0.18, 0.2, 0.22, 0.25]:
        pts = []
        row = 0
        y = d * np.sqrt(3) / 2
        while y <= 1.0:
            x_offset = 0 if row % 2 == 0 else d / 2
            x = x_offset
            while x <= 1.0:
                pts.append((x, y))
                x += d
            y += d * np.sqrt(3) / 2
            row += 1
        
        if len(pts) >= n_points_target:
            # Calculate potential radius for each point
            # Limited by distance to nearest neighbor and walls
            scored_pts = []
            for i, (px, py) in enumerate(pts):
                min_dist = min(px, 1-px, py, 1-py) # Wall constraint
                for j, (qx, qy) in enumerate(pts):
                    if i != j:
                        dist = np.sqrt((px-qx)**2 + (py-qy)**2)
                        min_dist = min(min_dist, dist/2)
                scored_pts.append((min_dist, px, py))
            
            # Sort by potential radius descending
            scored_pts.sort(key=lambda x: x[0], reverse=True)
            
            # Take top n_points_target
            candidate_pts = [(px, py) for _, px, py in scored_pts[:n_points_target]]
            
            # Check if this candidate set is "valid" in terms of spacing
            # We just want a good spatial distribution.
            if len(candidate_pts) == n_points_target:
                best_points = candidate_pts
                break # Found a good one? Or keep searching?
                # Let's just take the first good one for speed, or the best density.
                # Actually, let's generate multiple candidates and pick the one with max min-distance.
        
    # Fallback: Simple grid if hex fails
    if not best_points:
        side = int(np.ceil(np.sqrt(N)))
        best_points = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:N]
        
    return best_points[:N]

def calculate_score(coords_radii):
    """Calculate sum of radii. Returns 0 if invalid."""
    n = len(coords_radii)
    if n != N:
        return 0.0
    
    # Check bounds
    for x, y, r in coords_radii:
        if x < r - 1e-9 or x > 1 - r + 1e-9:
            return 0.0
        if y < r - 1e-9 or y > 1 - r + 1e-9:
            return 0.0
        if r < 0:
            return 0.0
            
    # Check overlaps
    for i in range(n):
        xi, yi, ri = coords_radii[i]
        for j in range(i+1, n):
            xj, yj, rj = coords_radii[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-9:
                return 0.0
                
    return sum(r for _, _, r in coords_radii)

def repair(coords_radii):
    """Attempt to repair a slightly invalid configuration."""
    coords_radii = np.array(coords_radii, dtype=float)
    n = len(coords_radii)
    
    # Simple iterative projection
    for _ in range(50):
        changed = False
        # Wall constraints
        for i in range(n):
            x, y, r = coords_radii[i]
            # If outside, move in and shrink if necessary
            if x < r:
                coords_radii[i, 0] = r
                changed = True
            if x > 1 - r:
                coords_radii[i, 0] = 1 - r
                changed = True
            if y < r:
                coords_radii[i, 1] = r
                changed = True
            if y > 1 - r:
                coords_radii[i, 1] = 1 - r
                changed = True
            
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = coords_radii[i]
                xj, yj, rj = coords_radii[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist:
                    # Push apart
                    if dist == 0:
                        # Random push
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dist = max(dist, 1e-8)
                    
                    # Move each center by half the overlap
                    overlap = min_dist - dist
                    push_x = (dx / dist) * overlap / 2
                    push_y = (dy / dist) * overlap / 2
                    
                    coords_radii[i, 0] -= push_x
                    coords_radii[i, 1] -= push_y
                    coords_radii[j, 0] += push_x
                    coords_radii[j, 1] += push_y
                    changed = True
                    
        if not changed:
            break
            
    return coords_radii

def run_slsqp(initial):
    """Run SLSQP optimization starting from initial configuration."""
    n = N
    # Vector format: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = initial[i][0]
        x0[3*i+1] = initial[i][1]
        # Initial radius: small fraction of min distance to neighbors
        min_d = 1.0
        for j in range(n):
            if i != j:
                d = np.sqrt((initial[i][0]-initial[j][0])**2 + (initial[i][1]-initial[j][1])**2)
                min_d = min(min_d, d)
        min_d = min(min_d, initial[i][0], 1-initial[i][0], initial[i][1], 1-initial[i][1])
        x0[3*i+2] = min_d / 2 * 0.9 # Slightly smaller to be safe

    bounds = [(0, 1), (0, 1), (0, 0.5)] * n

    # Constraints
    cons = []
    # Wall constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
        
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2) - (v[3*i+2] + v[3*j+2])
            })

    def objective(v):
        return -np.sum(v[2::3])

    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
        if res.success:
            return res.x
        else:
            # Return the last point even if not fully successful, we will repair it
            return res.x
    except Exception:
        return x0

def construct():
    global DEADLINE
    best_score = 0.0
    best_solution = None
    
    # Initial grid
    base_init = get_hex_grid()
    
    # Multi-start
    for restart in range(20):
        if time.time() > DEADLINE:
            break
            
        # Jitter the initial positions
        init = []
        for x, y in base_init:
            # Small random jitter
            x += np.random.uniform(-0.01, 0.01)
            y += np.random.uniform(-0.01, 0.01)
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            init.append((x, y))
            
        # Run optimizer
        result_vec = run_slsqp(init)
        
        # Reshape to (N, 3)
        sol = result_vec.reshape(N, 3)
        
        # Repair if necessary
        sol = repair(sol)
        
        # Evaluate
        score = calculate_score(sol)
        
        if score > best_score:
            best_score = score
            best_solution = sol
            
    # If no valid solution found, return the grid (which should be valid)
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution

# EVOLVE-BLOCK-END
