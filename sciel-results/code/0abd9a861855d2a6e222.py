import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill

# Global time tracking
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return DEADLINE - _elapsed()

def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    return (lo, hi)

def _enforce_validity(params):
    """
    Enforce all constraints: walls and pairwise non-overlap by shrinking radii.
    This is the safety net to ensure returned solutions are valid.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
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
            # Shrink the smaller circle by ov (or split if equal)
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
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p

def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))

def _physics_relax(xyz, time_limit=2.0, grow_rate=0.002, seed=42):
    """
    Force-directed physics relaxation that grows radii.
    Vectorized for speed.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = _score_valid(xyz)
    best_xyz = xyz.copy()
    
    dt = 0.002
    
    for iteration in range(20000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces (Vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx * dx + dy * dy
        np.fill_diagonal(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        overlap = r[:, None] + r[None, :] - dist
        overlap = np.maximum(overlap, 0)
        
        force_mag = overlap * 10.0
        inv_dist = 1.0 / dist
        fx_mat = force_mag * dx * inv_dist
        fy_mat = force_mag * dy * inv_dist
        
        fx = np.sum(fx_mat, axis=1)
        fy = np.sum(fy_mat, axis=1)
        
        # Wall forces
        # Left x
        wall_lx = np.maximum(0, r - x)
        fx -= wall_lx * 10.0
        # Right x
        wall_ux = np.maximum(0, x + r - 1)
        fx -= wall_ux * 10.0
        # Left y
        wall_ly = np.maximum(0, r - y)
        fy -= wall_ly * 10.0
        # Right y
        wall_uy = np.maximum(0, y + r - 1)
        fy -= wall_uy * 10.0
        
        # Update positions
        x += fx * dt
        y += fy * dt
        
        # Clip positions to [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Enforce validity strictly to prevent divergence
        current_xyz = np.column_stack((x, y, r))
        current_xyz = _enforce_validity(current_xyz)
        
        s = _score_valid(current_xyz)
        if s > best_score:
            best_score = s
            best_xyz = current_xyz.copy()
            
    return best_xyz

def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad

def _analytic_polish(p, time_limit=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B.
    """
    start_time = time.time()
    params = p.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    lo, hi = _bounds()
    bounds = list(zip(lo, hi))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_limit:
            break
            
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v
            
        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g
            
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14, 
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _grid_initial(seed):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr

def _hex_initial(seed):
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    p = points[indices]
    radii = np.full(N, 0.05)
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params

def _voronoi_initial(seed):
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    try:
        vor = Voronoi(points)
    except Exception:
        return _grid_initial(seed)
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params

def construct():
    global _start_time
    _start_time = time.time()
    
    best_params = None
    best_score = -np.inf
    
    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
    ]
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 11, 22]
    
    candidates = []
    
    # Phase 1: Physics Exploration
    for seed in seeds:
        if _remaining() < 15.0: # Reserve 15s for Polish
            break
            
        for init_fn in initializers:
            if _remaining() < 15.0:
                break
                
            try:
                x0 = init_fn(seed)
                x0 = _enforce_validity(x0).reshape(N, 3)
                
                # Run physics relaxation
                p_phys = _physics_relax(x0, time_limit=2.0, seed=seed)
                s = _score_valid(p_phys)
                
                candidates.append((s, p_phys, seed))
                
            except Exception:
                continue
                
    if not candidates:
        # Fallback
        p = _enforce_validity(_grid_initial(42)).reshape(N, 3)
        return p
        
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Analytic Polish on Top Candidates
    # Take top 5 candidates and refine them with L-BFGS-B
    top_k = 5
    for i in range(min(top_k, len(candidates))):
        if _remaining() < 5.0:
            break
        s, p, seed = candidates[i]
        
        # Run Analytic Polish
        p_polished = _analytic_polish(p, time_limit=min(_remaining() - 2.0, 8.0))
        s_polished = _score_valid(p_polished)
        
        if s_polished > best_score:
            best_score = s_polished
            best_params = p_polished.copy()
            
    # If no polish ran or failed, use best physics result
    if best_params is None:
        best_params = candidates[0][1]
        best_score = candidates[0][0]
        
    # Final safety check
    best_params = _enforce_validity(best_params)
    
    return best_params
