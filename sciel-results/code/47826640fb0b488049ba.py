"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145.0  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


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
    grad[1::3] += -2.0 * penalty_weight * sp_uy * sg_uy
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


def _objective_knn(params, penalty_weight, smooth_eps=1e-4, k_neighbors=8):
    """
    Vectorized objective using k-NN for overlap checking (faster than full pairwise).
    No gradient, used for initial broad search.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # k-NN overlaps
    centers = np.column_stack([x, y])
    try:
        tree = cKDTree(centers)
        # Query k nearest neighbors for each point
        dists, indices = tree.query(centers, k=min(k_neighbors+1, N))
        # dists[i][0] is 0 (self), so skip index 0
        for i in range(N):
            for j_idx in range(1, min(k_neighbors+1, N)):
                j = indices[i, j_idx]
                if j <= i:
                    continue  # avoid double-counting
                overlap = r[i] + r[j] - dists[i, j_idx]
                penalty += _softplus(overlap, smooth_eps)**2
    except Exception:
        # Fallback to full pairwise
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty


def _objective_full(params, penalty_weight, smooth_eps=1e-4):
    """
    Full pairwise objective for final polish (most accurate but slower).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Full pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty


def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True


def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz


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
    iu = np.triu_indices(N, k=1)
    for _ in range(300):
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


def _initial_hex():
    """Hexagonal packing initial config - fixed spacing."""
    positions = []
    r = 0.09
    # 5 columns, 6 rows with hex offset
    dx = 1.0 / 5.0
    dy = 1.0 / 6.0
    for row in range(6):
        for col in range(5):
            if row % 2 == 1:
                x = (col + 1.0) * dx
            else:
                x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config - 5x5 with center removed + extras."""
    positions = []
    r = 0.09
    # 5x5 grid = 25 positions, remove center, add 1 more
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    # Add one more circle in the center
    positions.append([0.5, 0.5, r * 0.85])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Structured config with corner/edge preference."""
    rng = np.random.default_rng(seed)
    positions = []
    # Corners with larger radius
    corners = [[0.07, 0.07], [0.93, 0.07], [0.07, 0.93], [0.93, 0.93]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.10])
    # Edge midpoints
    edges = [[0.5, 0.07], [0.5, 0.93], [0.07, 0.5], [0.93, 0.5],
             [0.25, 0.07], [0.75, 0.07], [0.25, 0.93], [0.75, 0.93],
             [0.07, 0.25], [0.07, 0.75], [0.93, 0.25], [0.93, 0.75]]
    for cx, cy in edges:
        positions.append([cx, cy, 0.085])
        if len(positions) >= N:
            break
    # Fill rest randomly in interior
    for _ in range(N - len(positions)):
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        positions.append([x, y, rng.uniform(0.07, 0.095)])
    return np.array(positions[:N])


def _initial_ring(seed=0):
    """Ring-shaped initial config with better spacing."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.09]]
    # Ring 1: 6 circles at radius ~0.18 from center
    for k in range(6):
        angle = 2*np.pi*k/6 + 0.1
        dist = 0.19
        x = 0.5 + dist*np.cos(angle)
        y = 0.5 + dist*np.sin(angle)
        x = np.clip(x, 0.08, 0.92)
        y = np.clip(y, 0.08, 0.92)
        positions.append([x, y, 0.09])
    # Ring 2: 10 circles at radius ~0.32 from center
    for k in range(10):
        angle = 2*np.pi*k/10 + 0.15
        dist = 0.33
        x = 0.5 + dist*np.cos(angle)
        y = 0.5 + dist*np.sin(angle)
        x = np.clip(x, 0.07, 0.93)
        y = np.clip(y, 0.07, 0.93)
        positions.append([x, y, 0.08])
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.07])
    return np.array(positions[:N])


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B with escalating penalty.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
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


def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: k-NN based with moderate penalty (fast, broad)
    remaining = deadline - time.time()
    if remaining > 6:
        penalty = 10.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: k-NN with higher penalty (tighten constraints)
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 50.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: Analytic gradient with high penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 200.0
        def obj3(p):
            v, _ = _objective_grad(p, penalty, 1e-5)
            return v
        def grad3(p):
            _, g = _objective_grad(p, penalty, 1e-5)
            return g
        try:
            res = minimize(obj3, params, jac=grad3, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-13})
            obj_val, _ = _objective_grad(res.x, penalty, 1e-5)
            if obj_val < _objective_grad(params, penalty, 1e-5)[0]:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty, analytic gradient
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 1000.0
        def obj4(p):
            v, _ = _objective_grad(p, penalty, 1e-6)
            return v
        def grad4(p):
            _, g = _objective_grad(p, penalty, 1e-6)
            return g
        try:
            res = minimize(obj4, params, jac=grad4, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            obj_val, _ = _objective_grad(res.x, penalty, 1e-6)
            if obj_val < _objective_grad(params, penalty, 1e-6)[0]:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish with analytic gradient
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 5000.0
        def obj5(p):
            v, _ = _objective_grad(p, penalty, 1e-7)
            return v
        def grad5(p):
            _, g = _objective_grad(p, penalty, 1e-7)
            return g
        try:
            res = minimize(obj5, params, jac=grad5, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 30})
            obj_val, _ = _objective_grad(res.x, penalty, 1e-7)
            if obj_val < _objective_grad(params, penalty, 1e-7)[0]:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 6: Ultra-precise final polish with smallest smoothness
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 10000.0
        def obj6(p):
            v, _ = _objective_grad(p, penalty, 1e-8)
            return v
        def grad6(p):
            _, g = _objective_grad(p, penalty, 1e-8)
            return g
        try:
            res = minimize(obj6, params, jac=grad6, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 40})
            obj_val, _ = _objective_grad(res.x, penalty, 1e-8)
            if obj_val < _objective_grad(params, penalty, 1e-8)[0]:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Final analytic polish pass
    remaining = deadline - time.time()
    if remaining > 1.0:
        xyz_polished = _polish_analytic(best_params.reshape(N, 3), time_budget=min(remaining - 0.5, 5.0))
        if _validate(xyz_polished):
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_params = xyz_polished.flatten()
                best_score = score
    
    return best_params.reshape(N, 3), best_score


def construct():
    global_start = time.time()
    deadline = global_start + _TIME_LIMIT  # explicit deadline
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_initial_hex())
    inits.append(_initial_corner_weighted(0))
    inits.append(_initial_corner_weighted(1))
    inits.append(_initial_corner_weighted(2))
    inits.append(_initial_ring(0))
    inits.append(_initial_ring(1))
    
    # Optimize from each, allocating more time to earlier (likely better) starts
    n_inits = len(inits)
    for idx, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 12.0:
            break
        # Allocate proportionally: first init gets more time
        time_frac = (n_inits - idx) / (n_inits * (n_inits + 1) / 2.0)
        time_budget = deadline - 3.0 - remaining * 0.1
        xyz_opt, score = _optimize_single(xyz0, time_budget)
        
        # Fix if invalid
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(100):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.95
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            # Try enforce_validity as last resort
            xyz_opt = _enforce_validity(xyz_opt.flatten())
            if not _validate(xyz_opt):
                continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
