"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p

def _physics_relax_grow(xyz, time_limit=15.0, grow_rate=0.0003):
    """
    Force-directed physics relaxation that grows radii over time.
    This is the primary optimization engine - it naturally handles
    circle packing constraints through repulsion forces and radius growth.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Track best valid configuration
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.003
    velocity_damp = 0.92
    
    for iteration in range(20000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly to push packing tighter
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) + 1e-10
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Strong repulsion proportional to overlap
                    force = (min_dist - dist) * 80.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces - push circles away from walls
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 150.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 150.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 150.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 150.0
        
        # Update positions with velocity damping
        x += dt * fx
        y += dt * fy
        
        # Clip to valid positions
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        # Check validity and track best
        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
    
    return best_xyz

def _objective_full(params, penalty_weight=1000.0):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    pen_x_lo = np.maximum(0, -(x - r))
    pen_x_hi = np.maximum(0, x + r - 1)
    pen_y_lo = np.maximum(0, -(y - r))
    pen_y_hi = np.maximum(0, y + r - 1)
    pen_r = np.maximum(0, -r)
    penalty += np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
    penalty += np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
    penalty += np.sum(pen_r ** 2)
    
    # Full pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + penalty_weight * penalty

def _polish_full(xyz, time_budget=10.0):
    """
    Final high-precision polish using full pairwise constraints.
    Uses very tight tolerances and high penalty weights for precision.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _objective_full(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _objective_full(p, penalty_weight)
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _initial_config_hex():
    """Hexagonal initial configuration for better packing density."""
    xyz = np.zeros((N, 3))
    r0 = 0.09
    row = 0
    col = 0
    spacing_x = 2 * r0 * 0.95
    spacing_y = 2 * r0 * 0.82  # hexagonal spacing
    
    count = 0
    while count < N and row < 10:
        x_offset = (row % 2) * spacing_x * 0.5
        x = x_offset + col * spacing_x
        y = spacing_y * 0.5 + row * spacing_y
        
        if x <= 1 - r0 and y <= 1 - r0:
            xyz[count, 0] = x
            xyz[count, 1] = y
            xyz[count, 2] = r0
            count += 1
        
        col += 1
        if col > 10:
            col = 0
            row += 1
    
    return xyz

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.7
            cells.append([x, y, r])
    return np.array(cells[:N])

def _local_refine_delaunay(xyz, time_budget=8.0):
    """
    Refine using L-BFGS-B with Delaunay-based constraints for speed.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _objective_full(best_params, 100.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    def obj_delaunay(p):
        n = N
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        pts = np.column_stack([x, y])
        edges = _get_delaunay_edges(pts)
        
        wall_penalty = 0.0
        for i in range(n):
            wall_penalty += np.maximum(0.0, r[i] - x[i])**2
            wall_penalty += np.maximum(0.0, r[i] - (1.0 - x[i]))**2
            wall_penalty += np.maximum(0.0, r[i] - y[i])**2
            wall_penalty += np.maximum(0.0, r[i] - (1.0 - y[i]))**2
        
        overlap_penalty = 0.0
        for i, j in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                overlap_penalty += overlap ** 2
        
        neg_penalty = np.sum(np.maximum(0.0, -r)**2)
        return -np.sum(r) + 100.0 * (wall_penalty + overlap_penalty + neg_penalty)
    
    try:
        result = minimize(obj_delaunay, best_params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        obj_val = _objective_full(result.x, 100.0)
        if obj_val < best_obj:
            best_params = result.x
            best_obj = obj_val
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # 1. Generate multiple initial configurations
    xyz_hex = _initial_config_hex()
    xyz_grid = _initial_config_grid()
    
    # 2. Physics relaxation with radius growth for each initial config
    remaining = deadline - (time.time() - global_start)
    if remaining < 5.0:
        return _enforce_validity(xyz_hex.flatten())
    
    # Use hexagonal for main physics run, grid as backup
    time_for_physics = min(remaining * 0.6, 60.0)
    xyz_physics_hex = _physics_relax_grow(xyz_hex, time_limit=time_for_physics)
    
    # Check time
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    
    # Also run physics on grid if time permits
    xyz_physics_grid = xyz_grid
    if remaining > 20.0:
        time_for_physics2 = min(remaining * 0.3, 20.0)
        xyz_physics_grid = _physics_relax_grow(xyz_grid, time_limit=time_for_physics2)
        remaining = deadline - (time.time() - global_start)
    
    # 3. Choose the better physics result
    score_hex = np.sum(xyz_physics_hex[:, 2])
    score_grid = np.sum(xyz_physics_grid[:, 2])
    xyz_best_physics = xyz_physics_hex if score_hex >= score_grid else xyz_physics_grid
    
    # 4. Local refinement with Delaunay constraints
    if remaining > 5.0:
        time_for_refine = min(remaining * 0.5, 15.0)
        xyz_refined = _local_refine_delaunay(xyz_best_physics, time_budget=time_for_refine)
    else:
        xyz_refined = xyz_best_physics
    
    remaining = deadline - (time.time() - global_start)
    
    # 5. Final full pairwise polish
    if remaining > 5.0:
        time_for_polish = min(remaining * 0.8, 20.0)
        xyz_polished = _polish_full(xyz_refined, time_budget=time_for_polish)
    else:
        xyz_polished = xyz_refined
    
    # 6. Enforce validity
    xyz_fixed = _enforce_validity(xyz_polished.flatten())
    
    # 7. Final check
    if not _validate(xyz_fixed):
        # Fallback to physics result
        xyz_fixed = _enforce_validity(xyz_best_physics.flatten())
    
    if not _validate(xyz_fixed):
        # Last resort: simple valid grid
        xyz_fixed = _enforce_validity(xyz_grid.flatten())
    
    return xyz_fixed

# EVOLVE-BLOCK-END
