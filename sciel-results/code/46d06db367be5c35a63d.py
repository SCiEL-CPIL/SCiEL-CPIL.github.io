import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed(start_time):
    return time.time() - start_time


def _remaining(start_time):
    return DEADLINE - _elapsed(start_time)


def _softplus(x, eps=1e-4):
    """Numerically stable softplus."""
    return eps * np.log1p(np.exp(np.clip(x / eps, -50, 50)))


def _softplus_grad(x, eps=1e-4):
    """Derivative of softplus: sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x / eps, -50, 50)))


def _validate(p):
    """Check if configuration p (N,3) is valid: all circles inside [0,1]^2 and non-overlapping."""
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    # Check walls
    if np.any(x < -1e-8) or np.any(x > 1 + 1e-8):
        return False
    if np.any(y < -1e-8) or np.any(y > 1 + 1e-8):
        return False
    if np.any(r < -1e-8):
        return False
    
    # Check wall containment
    if np.any(x - r < -1e-8) or np.any(x + r > 1 + 1e-8):
        return False
    if np.any(y - r < -1e-8) or np.any(y + r > 1 + 1e-8):
        return False
    
    # Check pairwise non-overlap
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any((r[iu[0]] + r[iu[1]]) > dist + 1e-8):
        return False
    
    return True


def _fix_boundaries(p):
    """Fix circles that are outside the unit square or have negative radii."""
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p


def _objective_penalty(params, penalty=1000.0, eps=1e-4):
    """Penalty objective for circle packing."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    score = -np.sum(r)
    
    # Wall violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)
    
    # Pairwise overlap violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy + 1e-10)
    overlap = (r[iu[0]] + r[iu[1]]) - dist
    pair = np.sum(_softplus(overlap, eps))
    
    return score + penalty * (pair + wall)


def _objective_grad(params, penalty=1000.0, eps=1e-4):
    """Gradient of penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    g = np.zeros_like(params)
    
    # d(-sum r)/dr = -1
    g[2::3] = -1.0
    
    # Wall contributions
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
    
    # Pairwise - smooth gradient
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy + 1e-10)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    sig = _softplus_grad(ov, eps)
    weight = penalty * sig / dist_safe
    
    for k in range(len(iu[0])):
        i = iu[0][k]
        j = iu[1][k]
        w = weight[k]
        adx = dx[k]
        ady = dy[k]
        g[3*i] += -w * adx
        g[3*j] += +w * adx
        g[3*i+1] += -w * ady
        g[3*j+1] += +w * ady
        g[3*i+2] += penalty * sig[k]
        g[3*j+2] += penalty * sig[k]
    
    return g


def _get_contact_edges(p, tol=1e-4):
    """Get pairs of circles that are nearly touching (within tol)."""
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    contact = np.abs((r[iu[0]] + r[iu[1]]) - dist) < tol
    return list(zip(iu[0][contact], iu[1][contact]))


def _initial_hex(seed):
    """Hexagonal packing initialization."""
    rng = np.random.RandomState(seed)
    
    # Generate hexagonal lattice points
    points = []
    spacing = 0.18
    for i in range(12):
        for j in range(12):
            x = i * spacing
            y = j * spacing * 1.732 + (0.5 * spacing * 1.732 if i % 2 else 0)
            points.append((x, y))
    
    points = np.array(points)
    
    # Center and scale to fit in [0.1, 0.9]^2
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.4 / max_coord
    points *= scale
    points += 0.5
    
    # Select N closest to center
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    p = points[indices]
    
    # Estimate radii based on local density
    radii = np.zeros(N)
    for i in range(N):
        dists_to_others = np.linalg.norm(p - p[i], axis=1)
        dists_to_others[i] = np.inf
        min_dist = np.min(dists_to_others)
        wall_dist = np.min([p[i,0], 1-p[i,0], p[i,1], 1-p[i,1]])
        radii[i] = 0.45 * np.min([min_dist, wall_dist])
    
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _initial_grid(seed):
    """Grid initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((x, y, r))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _initial_corner_weighted(seed):
    """Initialization with circles biased toward corners and edges."""
    rng = np.random.RandomState(seed)
    
    # Place some circles near corners and edges
    positions = []
    # 4 corners
    positions.append((0.08, 0.08))
    positions.append((0.92, 0.08))
    positions.append((0.08, 0.92))
    positions.append((0.92, 0.92))
    
    # 4 edge midpoints
    positions.append((0.5, 0.08))
    positions.append((0.5, 0.92))
    positions.append((0.08, 0.5))
    positions.append((0.92, 0.5))
    
    # Remaining circles randomly
    while len(positions) < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        positions.append((x, y))
    
    positions = np.array(positions[:N])
    
    # Estimate radii
    radii = np.zeros(N)
    for i in range(N):
        dists_to_others = np.linalg.norm(positions - positions[i], axis=1)
        dists_to_others[i] = np.inf
        min_dist = np.min(dists_to_others)
        wall_dist = np.min([positions[i,0], 1-positions[i,0], positions[i,1], 1-positions[i,1]])
        radii[i] = 0.45 * np.min([min_dist, wall_dist])
    
    params = np.empty(3 * N)
    params[0::3] = positions[:, 0]
    params[1::3] = positions[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clamp to walls
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
        
        max_ov = 0.0
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            max_ov = max(max_ov, ov)
            # Shrink the smaller circle
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov * 0.5, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                shrink2 = min(ov * 0.5, p[j, 2] - 1e-6)
                p[j, 2] -= shrink2
            else:
                shrink = min(ov * 0.5, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                shrink2 = min(ov * 0.5, p[i, 2] - 1e-6)
                p[i, 2] -= shrink2
        
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
        if max_ov < 1e-8:
            break
    
    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _lbfgs_optimize(p, penalty=1000.0, maxiter=1000, time_limit=5.0, eps=1e-4):
    """L-BFGS-B optimization with smooth penalty."""
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    bounds_list = list(zip(lo, hi))
    
    current = p.flatten().copy()
    
    # Adaptive penalty annealing
    penalties = [5000.0, 2000.0, 1000.0, 500.0, 200.0, 100.0]
    smoothnesses = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
    
    for idx, (pen, sm) in enumerate(zip(penalties, smoothnesses)):
        if time.time() - start_time > time_limit - 0.3:
            break
        
        def obj(params):
            return _objective_penalty(params, penalty=pen, eps=sm)
        
        def grad(params):
            return _objective_grad(params, penalty=pen, eps=sm)
        
        try:
            res = minimize(
                obj,
                current,
                method='L-BFGS-B',
                jac=grad,
                bounds=bounds_list,
                options={'maxiter': maxiter // len(penalties), 'ftol': 1e-14, 'gtol': 1e-10}
            )
            current = res.x
            p_new = _enforce_validity(current.reshape(N, 3))
            s_new = _score_valid(p_new)
            if s_new > best_score:
                best_score = s_new
                best_p = p_new.copy()
        except Exception:
            continue
    
    return best_p, best_score


def _physics_shake(p, time_limit=10.0, seed=42):
    """
    Physics-based shake: treat circles as particles with repulsive forces.
    Simulate brief dynamics to escape local minima.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    # Initialize velocities
    vel = rng.normal(0, 0.01, size=(N, 2))
    
    dt = 0.001
    damping = 0.95
    repulsion_strength = 50.0
    wall_repulsion = 100.0
    
    steps = int(time_limit / (dt * 20))  # ~50 steps per second of real time
    
    for step in range(steps):
        if time.time() - start > time_limit:
            break
        
        x = p[:, 0].copy()
        y = p[:, 1].copy()
        r = p[:, 2].copy()
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy + 1e-10)
        overlap = (r[iu[0]] + r[iu[1]]) - dist
        
        for k in range(len(iu[0])):
            if overlap[k] > 0:
                force = repulsion_strength * overlap[k] / dist[k]
                fx[iu[0][k]] += force * dx[k]
                fy[iu[0][k]] += force * dy[k]
                fx[iu[1][k]] -= force * dx[k]
                fy[iu[1][k]] -= force * dy[k]
        
        # Wall repulsion
        for i in range(N):
            if x[i] - r[i] < 0:
                fx[i] += wall_repulsion * (r[i] - x[i])
            if x[i] + r[i] > 1:
                fx[i] -= wall_repulsion * (x[i] + r[i] - 1)
            if y[i] - r[i] < 0:
                fy[i] += wall_repulsion * (r[i] - y[i])
            if y[i] + r[i] > 1:
                fy[i] -= wall_repulsion * (y[i] + r[i] - 1)
        
        # Update velocities and positions
        vel += np.column_stack([fx, fy]) * dt
        vel *= damping
        x += vel[:, 0] * dt
        y += vel[:, 1] * dt
        
        # Clamp to walls
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        p[:, 0] = x
        p[:, 1] = y
        
        # Periodically enforce validity and check score
        if step % 10 == 0:
            p_valid = _enforce_validity(p)
            s_valid = _score_valid(p_valid)
            if s_valid > best_score:
                best_score = s_valid
                best_p = p_valid.copy()
    
    return best_p, best_score


def _slsqp_optimize(p, time_limit=5.0):
    """SLSQP optimization with explicit constraints."""
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    x0 = p.flatten().copy()
    
    # Define constraints
    def wall_constraint(i, axis):
        def con(params):
            p = params.reshape(N, 3)
            if axis == 0:
                return np.minimum(p[i, 0] - p[i, 2], p[i, 0] + p[i, 2] - 1.0)
            else:
                return np.minimum(p[i, 1] - p[i, 2], p[i, 1] + p[i, 2] - 1.0)
        return con
    
    def pair_constraint(i, j):
        def con(params):
            p = params.reshape(N, 3)
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            return dist - p[i, 2] - p[j, 2]
        return con
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': wall_constraint(i, 0)})
        constraints.append({'type': 'ineq', 'fun': wall_constraint(i, 1)})
    
    iu = np.triu_indices(N, k=1)
    for i, j in zip(iu[0], iu[1]):
        constraints.append({'type': 'ineq', 'fun': pair_constraint(i, j)})
    
    def obj(params):
        p = params.reshape(N, 3)
        return -np.sum(p[:, 2])
    
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    bounds = list(zip(lo, hi))
    
    try:
        res = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.success:
            p_new = _enforce_validity(res.x.reshape(N, 3))
            s_new = _score_valid(p_new)
            if s_new > best_score:
                best_score = s_new
                best_p = p_new.copy()
    except Exception:
        pass
    
    return best_p, best_score


def construct():
    start_time = time.time()
    
    best_params = None
    best_score = -np.inf
    
    initializers = [
        _initial_hex,
        _initial_grid,
        _initial_corner_weighted,
    ]
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100,
             1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]
    
    # Phase 1: Multi-start L-BFGS
    for seed in seeds:
        if _remaining(start_time) < 25:
            break
        for init_fn in initializers:
            if _remaining(start_time) < 25:
                break
            
            x0 = init_fn(seed)
            p0 = _enforce_validity(x0)
            
            try:
                p, s = _lbfgs_optimize(p0, penalty=1000.0, maxiter=500, 
                                       time_limit=min(3.0, max(0.5, _remaining(start_time) - 30)))
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
            except Exception:
                continue
    
    if best_params is None:
        best_params = _enforce_validity(_initial_grid(42))
        best_score = _score_valid(best_params)
    
    # Phase 2: Physics-based shake for escape
    if _remaining(start_time) > 15:
        try:
            p_phys, s_phys = _physics_shake(
                best_params,
                time_limit=min(10.0, max(1.0, _remaining(start_time) - 20)),
                seed=42
            )
            if s_phys > best_score:
                best_score = s_phys
                best_params = p_phys
        except Exception:
            pass
    
    # Phase 3: SLSQP refinement
    if _remaining(start_time) > 10:
        try:
            p_slsqp, s_slsqp = _slsqp_optimize(
                best_params,
                time_limit=min(5.0, max(1.0, _remaining(start_time) - 15))
            )
            if s_slsqp > best_score:
                best_score = s_slsqp
                best_params = p_slsqp
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS polish
    if _remaining(start_time) > 5:
        try:
            p_final, s_final = _lbfgs_optimize(
                best_params,
                penalty=2000.0,
                maxiter=300,
                time_limit=min(4.0, max(0.5, _remaining(start_time) - 8)),
                eps=1e-5
            )
            if s_final > best_score:
                best_score = s_final
                best_params = p_final
        except Exception:
            pass
    
    # Final validation
    best_params = _enforce_validity(best_params)
    
    return best_params.copy()
