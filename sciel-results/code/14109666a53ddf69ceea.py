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

def _physics_relax(xyz, time_limit=3.0, grow_rate=0.001):
    """
    Force-directed physics relaxation that also grows radii over time.
    Uses adaptive step size and smooth repulsion potential for stability.
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
    max_dt = 0.01
    min_dt = 0.0005
    
    # Adaptive growth: start fast, slow down
    total_iterations = 6000
    for iteration in range(total_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive growth rate: faster early, slower late
        progress = iteration / total_iterations
        adaptive_grow = grow_rate * (1.0 + 2.0 * (1.0 - progress))
        r *= (1.0 + adaptive_grow)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces using smooth repulsion
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion with smooth potential
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) + 1e-10
                min_dist = r[i] + r[j]
                
                overlap = min_dist - dist
                if overlap > 0:
                    # Smooth quadratic repulsion: force ~ overlap^2 for stability
                    force = overlap * 80.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                else:
                    # Mild attraction when not overlapping to encourage packing
                    gap = -overlap
                    if gap < 0.05:  # Only within 0.05 of touching
                        force = -gap * 5.0
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
        
        # Boundary forces - smooth quadratic
        for i in range(N):
            if x[i] < r[i]:
                pen = r[i] - x[i]
                fx[i] += pen * 150.0
            if x[i] > 1 - r[i]:
                pen = x[i] - (1 - r[i])
                fx[i] -= pen * 150.0
            if y[i] < r[i]:
                pen = r[i] - y[i]
                fy[i] += pen * 150.0
            if y[i] > 1 - r[i]:
                pen = y[i] - (1 - r[i])
                fy[i] -= pen * 150.0
        
        # Adaptive step size: reduce if forces are large
        max_force = np.max(np.sqrt(fx**2 + fy**2))
        if max_force > 10.0:
            dt = max(min_dt, dt * 0.7)
        elif max_force < 0.5:
            dt = min(max_dt, dt * 1.1)
        
        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Clip to valid positions
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        # Check validity and track best (less frequent for speed)
        if iteration % 10 == 0:
            if _validate(xyz):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_xyz = xyz.copy()
    
    # Final check
    if _validate(xyz):
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xyz = xyz.copy()
    
    return best_xyz

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

def _objective_full(params, penalty_weight=100.0, smooth_beta=50.0):
    """Full pairwise objective with smooth soft-plus penalties for better gradient behavior."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Smooth soft-plus: f(z) = log(1 + exp(beta*z))/beta, approximates max(0,z)
    def _softplus(z, beta=smooth_beta):
        # Numerically stable softplus
        z_beta = beta * z
        # For large positive z_beta, softplus ~ z; for large negative, ~ 0
        return np.where(z_beta > 20, z, np.log1p(np.exp(z_beta)) / beta)
    
    # Boundary violations (smooth)
    pen_x_lo = _softplus(-(x - r))
    pen_x_hi = _softplus(x + r - 1)
    pen_y_lo = _softplus(-(y - r))
    pen_y_hi = _softplus(y + r - 1)
    pen_r = _softplus(-r)
    
    penalty = np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
    penalty += np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
    penalty += np.sum(pen_r ** 2)
    
    # Full pairwise overlap penalties (smooth)
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-10
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap) ** 2
    
    return -np.sum(r) + penalty_weight * penalty

def _polish_full(xyz, time_budget=5.0):
    """
    Final high-precision polish using full pairwise constraints.
    Uses smooth soft-plus penalties and adaptive penalty weight schedule.
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
    
    # Progressive penalty weight increase with smoother transitions
    penalty_schedule = [500.0, 2000.0, 8000.0, 30000.0]
    beta_schedule = [20.0, 50.0, 100.0, 200.0]  # Smoother -> sharper
    
    for pw, beta in zip(penalty_schedule, beta_schedule):
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _objective_full(p, pw, beta)
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 8000})
            obj_val = obj(result.x)
            if obj_val < best_obj:
                best_params = result.x
                best_obj = obj_val
        except Exception:
            pass
    
    # Final high-precision pass with very high penalty
    if time.time() - start_time < time_budget - 0.5:
        def obj_final(p):
            return _objective_full(p, 100000.0, 500.0)
        
        try:
            result = minimize(obj_final, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 10000})
            obj_val = obj_final(result.x)
            if obj_val < best_obj:
                best_params = result.x
                best_obj = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.85
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.85
            positions.append([x, y, r])
    positions.append([0.5, 0.5, 0.5 / side * 0.4])
    return np.array(positions[:N])

def _initial_config_voronoi(seed):
    """Initialize using random points with estimated radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.08, 0.92, size=(N, 2))
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1 - p[0], p[1], 1 - p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.85
        
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    return params.reshape(N, 3)

def _initial_config_corner_weighted():
    """Corner-weighted configuration for better packing."""
    positions = []
    # Corners - larger circles
    for cx, cy in [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]:
        positions.append([cx, cy, 0.11])
    # Edge midpoints
    for cx, cy in [(0.5, 0.10), (0.5, 0.90), (0.10, 0.5), (0.90, 0.5)]:
        positions.append([cx, cy, 0.095])
    # Center region - ring
    for k in range(10):
        angle = 2 * np.pi * k / 10
        x = 0.5 + 0.14 * np.cos(angle)
        y = 0.5 + 0.14 * np.sin(angle)
        positions.append([x, y, 0.075])
    for k in range(8):
        angle = 2 * np.pi * k / 8 + np.pi / 8
        x = 0.5 + 0.26 * np.cos(angle)
        y = 0.5 + 0.26 * np.sin(angle)
        positions.append([x, y, 0.06])
    
    return np.array(positions[:N])

def _optimize_physics(xyz, time_budget=10.0):
    """
    Primary optimization: physics-based relaxation with growing radii.
    Uses adaptive time allocation between physics phases and polish.
    """
    start_time = time.time()
    xyz = xyz.copy()
    
    # Phase 1: Physics relaxation with moderate growth (40% of time)
    phys_time1 = min(4.0, time_budget * 0.4)
    if time_budget > 3.0 and phys_time1 > 1.0:
        xyz_phys = _physics_relax(xyz, time_limit=phys_time1, grow_rate=0.0010)
        if np.sum(xyz_phys[:, 2]) > np.sum(xyz[:, 2]):
            xyz = xyz_phys
    
    # Phase 2: More aggressive physics with higher growth (30% of time)
    remaining = time_budget - (time.time() - start_time)
    phys_time2 = min(3.0, remaining * 0.5)
    if remaining > 2.0 and phys_time2 > 1.0:
        xyz_phys2 = _physics_relax(xyz, time_limit=phys_time2, grow_rate=0.0018)
        if np.sum(xyz_phys2[:, 2]) > np.sum(xyz[:, 2]):
            xyz = xyz_phys2
    
    # Phase 3: Final polish with L-BFGS-B (remaining time)
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1.5:
        try:
            xyz_polished = _polish_full(xyz, time_budget=min(remaining * 0.85, 5.0))
            xyz_fixed = _enforce_validity(xyz_polished.flatten())
            if _validate(xyz_fixed) and np.sum(xyz_fixed[:, 2]) > np.sum(xyz[:, 2]):
                xyz = xyz_fixed
        except Exception:
            pass
    
    return xyz

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Voronoi-based random (multiple seeds)
    for seed in [42, 123, 456, 789, 1000, 2024]:
        initial_configs.append(_initial_config_voronoi(seed))
    
    # 4. Corner-weighted
    initial_configs.append(_initial_config_corner_weighted())
    
    # 5. Random configurations
    for trial in range(3):
        rng = np.random.RandomState(100 + trial)
        x = rng.uniform(0.15, 0.85, N)
        y = rng.uniform(0.15, 0.85, N)
        r = rng.uniform(0.06, 0.14, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    best_xyz = initial_configs[0]
    best_score = np.sum(best_xyz[:, 2])
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 6.0:
            break
        
        time_budget = min(15.0, remaining - 3.0)
        if time_budget < 3.0:
            break
        
        try:
            xyz_opt = _optimize_physics(xyz0, time_budget=time_budget)
            xyz_fixed = _enforce_validity(xyz_opt.flatten())
            
            if _validate(xyz_fixed):
                score = np.sum(xyz_fixed[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_fixed.copy()
        except Exception:
            pass
    
    # Final validation and polish
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz.flatten())
    
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 2.5 and _validate(best_xyz):
        try:
            # Use remaining time for final high-precision polish
            polish_time = min(remaining * 0.75, 10.0)
            polished = _polish_full(best_xyz, time_budget=polish_time)
            polished_valid = _enforce_validity(polished.flatten())
            if _validate(polished_valid) and np.sum(polished_valid[:, 2]) > np.sum(best_xyz[:, 2]):
                best_xyz = polished_valid.copy()
        except Exception:
            pass
    
    # Final safety check
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz.flatten())
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
