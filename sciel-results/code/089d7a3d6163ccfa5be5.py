"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _physics_relax(xy, r, deadline, max_steps=20000):
    """
    Physics-based relaxation:
    1. Apply repulsive forces between overlapping circles and walls.
    2. Integrate positions to resolve overlaps.
    3. Once stable, inflate radii and repeat.
    Returns (xy, r, total, valid).
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    # Start with small radii to ensure feasibility
    r = np.maximum(r, 0.01)
    
    # Initial relaxation to remove overlaps
    dt = 0.001
    damping = 0.95
    repulsion_strength = 50.0
    wall_strength = 100.0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        forces = np.zeros_like(xy)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Force proportional to overlap
                    f = repulsion_strength * overlap
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            overlap = r[i] - xy[i, 0]
            if overlap > 0:
                forces[i, 0] += wall_strength * overlap
            # Right wall
            overlap = r[i] - (1.0 - xy[i, 0])
            if overlap > 0:
                forces[i, 0] -= wall_strength * overlap
            # Bottom wall
            overlap = r[i] - xy[i, 1]
            if overlap > 0:
                forces[i, 1] += wall_strength * overlap
            # Top wall
            overlap = r[i] - (1.0 - xy[i, 1])
            if overlap > 0:
                forces[i, 1] -= wall_strength * overlap
        
        # Integrate
        xy += dt * forces
        xy *= damping  # Damping for stability
        
        # Clip to bounds
        for i in range(n):
            xy[i, 0] = np.clip(xy[i, 0], r[i], 1.0 - r[i])
            xy[i, 1] = np.clip(xy[i, 1], r[i], 1.0 - r[i])
        
        # Check if converged (max force small)
        max_force = np.max(np.abs(forces))
        if max_force < 1e-6:
            break
    
    # Now inflate radii iteratively
    inflation_factor = 1.02
    max_inflations = 200
    
    for infl in range(max_inflations):
        if time.time() > deadline:
            break
        
        # Try to inflate all radii by inflation_factor
        new_r = r * inflation_factor
        
        # Check if this is feasible (simple check)
        feasible = True
        for i in range(n):
            if new_r[i] > xy[i, 0] or new_r[i] > 1.0 - xy[i, 0] or new_r[i] > xy[i, 1] or new_r[i] > 1.0 - xy[i, 1]:
                feasible = False
                break
        if feasible:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_r[i] + new_r[j] - 1e-10:
                        feasible = False
                        break
                if not feasible:
                    break
        
        if not feasible:
            # Reduce inflation
            inflation_factor *= 0.9
            if inflation_factor < 1.001:
                break
            continue
        
        # Accept inflation and relax
        r = new_r
        xy = xy.copy()
        
        # Relax again
        dt = 0.0005
        for step in range(1000):
            if time.time() > deadline:
                break
            
            forces = np.zeros_like(xy)
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        f = repulsion_strength * overlap
                        fx = f * dx / dist
                        fy = f * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            for i in range(n):
                overlap = r[i] - xy[i, 0]
                if overlap > 0:
                    forces[i, 0] += wall_strength * overlap
                overlap = r[i] - (1.0 - xy[i, 0])
                if overlap > 0:
                    forces[i, 0] -= wall_strength * overlap
                overlap = r[i] - xy[i, 1]
                if overlap > 0:
                    forces[i, 1] += wall_strength * overlap
                overlap = r[i] - (1.0 - xy[i, 1])
                if overlap > 0:
                    forces[i, 1] -= wall_strength * overlap
            
            xy += dt * forces
            xy *= damping
            
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1.0 - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1.0 - r[i])
            
            max_force = np.max(np.abs(forces))
            if max_force < 1e-7:
                break
    
    # Final validation
    total, valid = _eval_packing(xy, r)
    if valid:
        return xy, r, total, True
    else:
        # Fallback to last valid state
        return xy, r, 0.0, False


def _coordinate_descent_polish(xy, r, deadline, max_sweeps=50):
    """
    Fast coordinate descent: for each circle, find the best position
    that maximizes its radius given all others fixed.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    if not best_valid:
        return best_xy, best_r, 0.0, False
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Find best position for circle i
            best_new_r = r[i]
            best_new_x = xy[i, 0]
            best_new_y = xy[i, 1]
            
            # Sample candidate positions
            for dx in np.linspace(-0.05, 0.05, 11):
                for dy in np.linspace(-0.05, 0.05, 11):
                    new_x = np.clip(xy[i, 0] + dx, 0.01, 0.99)
                    new_y = np.clip(xy[i, 1] + dy, 0.01, 0.99)
                    
                    # Max radius at this position
                    wall_clear = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
                    circle_clear = wall_clear
                    for j in range(n):
                        if j == i:
                            continue
                        dx2 = new_x - xy[j, 0]
                        dy2 = new_y - xy[j, 1]
                        dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        circle_clear = min(circle_clear, dist - r[j])
                    
                    new_r = max(0.001, circle_clear)
                    if new_r > best_new_r + 1e-10:
                        best_new_r = new_r
                        best_new_x = new_x
                        best_new_y = new_y
            
            if best_new_r > r[i] + 1e-10:
                xy[i, 0] = best_new_x
                xy[i, 1] = best_new_y
                r[i] = best_new_r
                improved = True
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total, best_valid


def _hex_init():
    """Hexagonal lattice initialization."""
    positions = []
    spacing = 0.15
    y = spacing / 2
    row = 0
    while len(positions) < N:
        x_offset = 0 if row % 2 == 0 else spacing * np.sqrt(3) / 4
        x = x_offset + spacing / 2
        while x < 1.0 - spacing / 2 and len(positions) < N:
            if x >= spacing / 2 and x <= 1.0 - spacing / 2 and y >= spacing / 2 and y <= 1.0 - spacing / 2:
                r = spacing * np.sqrt(3) / 4 - 0.001
                positions.append([x, y, max(r, 0.01)])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.02])
    
    return np.array(positions[:N])


def _random_init(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(N, 2))
    r = rng.uniform(0.02, 0.04, size=N)
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Phase 1: Multiple initializations with physics relaxation
    initializations = []
    
    # Hexagonal
    hex_pos = _hex_init()
    initializations.append((hex_pos[:, :2], hex_pos[:, 2]))
    
    # Random seeds
    for seed in range(10):
        xy, r = _random_init(seed=seed)
        initializations.append((xy, r))
    
    # Grid
    side = 5
    for i in range(side):
        for j in range(side):
            if len(initializations) >= 15:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r_val = 0.5 / side - 0.002
            if x >= 0.01 and x <= 0.99 and y >= 0.01 and y <= 0.99:
                xy = np.zeros((N, 2))
                r_arr = np.zeros(N)
                for k in range(N):
                    gi = k % side
                    gj = k // side
                    xy[k, 0] = (gi + 0.5) / side
                    xy[k, 1] = (gj + 0.5) / side
                    r_arr[k] = 0.5 / side - 0.002
                initializations.append((xy, r_arr))
                break
        if len(initializations) >= 15:
            break
    
    for init_idx, (xy, r) in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Physics relaxation
        out_xy, out_r, total, valid = _physics_relax(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
        
        # Coordinate descent polish
        if time.time() < deadline - 10:
            cd_xy, cd_r, cd_total, cd_valid = _coordinate_descent_polish(
                out_xy, out_r, deadline)
            if cd_valid and cd_total > best_total:
                best_total = cd_total
                best_valid = True
                best_xy = cd_xy
                best_r = cd_r
    
    # Phase 2: Final polish on best result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 5:
            final_xy, final_r, final_total, final_valid = _coordinate_descent_polish(
                best_xy, best_r, deadline, max_sweeps=30)
            if final_valid and final_total > best_total:
                best_total = final_total
                best_xy = final_xy
                best_r = final_r
    
    # Fallback
    if best_xy is None or not best_valid:
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
