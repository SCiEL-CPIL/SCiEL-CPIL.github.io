import numpy as np
import time
from scipy.optimize import minimize

N = 26
EPS = 1e-9
DEADLINE = 145.0  # safety margin before 150s kill


def _validate_and_score(params):
    """Validate configuration and return (is_valid, score)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    # Check walls
    if np.any(x < -EPS) or np.any(x > 1 + EPS):
        return False, 0.0
    if np.any(y < -EPS) or np.any(y > 1 + EPS):
        return False, 0.0
    if np.any(r < -EPS):
        return False, 0.0

    # Check circle fits in square
    if np.any(r - x < -EPS) or np.any(r - (1 - x) < -EPS):
        return False, 0.0
    if np.any(r - y < -EPS) or np.any(r - (1 - y) < -EPS):
        return False, 0.0

    # Check pairwise distances
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < (r[iu[0]] + r[iu[1]]) - EPS):
        return False, 0.0

    return True, np.sum(r)


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

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(100):
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


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
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
    """Hexagonal grid initialization."""
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
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        # Distance to nearest neighbor
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        
        # Distance to walls
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        
        # Initial radius is half the min distance, with a small safety factor
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
        
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    # Small jitter to break symmetry
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _pbd_refine(p, time_limit=15.0, seed=42):
    """
    Position Based Dynamics refinement.
    Treats circles as particles with mass proportional to radius^2.
    Uses Gauss-Seidel relaxation with adaptive outward pressure to maximize sum of radii.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # PBD parameters
    pressure = 0.5
    iterations_per_step = 50
    pressure_increase = 1.001
    pressure_max = 5.0
    
    current_pressure = pressure
    stagnation_count = 0
    max_stagnation = 100
    
    while time.time() - start < time_limit:
        p_work = best_p.copy()
        
        # Apply pressure: try to increase radii
        for i in range(N):
            p_work[i, 2] *= (1.0 + 0.001 * current_pressure)
            wall_dist = min(p_work[i, 0], 1.0 - p_work[i, 0],
                           p_work[i, 1], 1.0 - p_work[i, 1])
            p_work[i, 2] = min(p_work[i, 2], max(wall_dist, 1e-6))
        
        # Gauss-Seidel relaxation to resolve overlaps
        for _ in range(iterations_per_step):
            # Resolve wall constraints
            for i in range(N):
                if p_work[i, 0] < p_work[i, 2]:
                    p_work[i, 0] = p_work[i, 2]
                if p_work[i, 0] > 1.0 - p_work[i, 2]:
                    p_work[i, 0] = 1.0 - p_work[i, 2]
                if p_work[i, 1] < p_work[i, 2]:
                    p_work[i, 1] = p_work[i, 2]
                if p_work[i, 1] > 1.0 - p_work[i, 2]:
                    p_work[i, 1] = 1.0 - p_work[i, 2]
            
            # Resolve pairwise overlaps
            iu = np.triu_indices(N, k=1)
            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                dx = p_work[i, 0] - p_work[j, 0]
                dy = p_work[i, 1] - p_work[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = p_work[i, 2] + p_work[j, 2]
                
                if dist < min_dist:
                    if dist < 1e-10:
                        angle = rng.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = min_dist - dist
                    mi = p_work[i, 2] ** 2
                    mj = p_work[j, 2] ** 2
                    total_mass = mi + mj
                    
                    if total_mass < 1e-10:
                        total_mass = 1.0
                        mi = mj = 0.5
                    
                    corr_i = overlap * (mj / total_mass)
                    corr_j = overlap * (mi / total_mass)
                    
                    p_work[i, 0] += dx * corr_i
                    p_work[i, 1] += dy * corr_i
                    p_work[j, 0] -= dx * corr_j
                    p_work[j, 1] -= dy * corr_j
                    
                    p_work[i, 0] = np.clip(p_work[i, 0], 0.0, 1.0)
                    p_work[i, 1] = np.clip(p_work[i, 1], 0.0, 1.0)
                    p_work[j, 0] = np.clip(p_work[j, 0], 0.0, 1.0)
                    p_work[j, 1] = np.clip(p_work[j, 1], 0.0, 1.0)
            
            # Re-clamp radii to wall distances
            for i in range(N):
                wall_dist = min(p_work[i, 0], 1.0 - p_work[i, 0],
                               p_work[i, 1], 1.0 - p_work[i, 1])
                p_work[i, 2] = min(p_work[i, 2], max(wall_dist, 1e-6))
        
        p_work = _enforce_validity(p_work)
        s_new = _score_valid(p_work)
        
        if s_new > best_score + 1e-10:
            best_score = s_new
            best_p = p_work.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        current_pressure = min(current_pressure * pressure_increase, pressure_max)
        
        if stagnation_count > max_stagnation:
            p_work = best_p.copy()
            p_work[:, :2] += rng.normal(0, 0.005, size=(N, 2))
            p_work = _enforce_validity(p_work)
            best_p = p_work
            best_score = _score_valid(p_work)
            stagnation_count = 0
    
    return best_p


def _slsqp_refine(p, time_limit=10.0, seed=42):
    """
    Iterative SLSQP refinement.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    
    for iteration in range(20):
        if time.time() > start + time_limit:
            break
        
        x = best_p[:, 0]
        y = best_p[:, 1]
        r = best_p[:, 2]
        
        def obj(params):
            return -np.sum(params[2::3])
            
        def obj_grad(params):
            g = np.zeros_like(params)
            g[2::3] = -1.0
            return g
            
        cons = []
        
        for i in range(N):
            def wall_x1(params, i=i):
                return params[3*i] - params[3*i+2]
            def wall_x1_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i] = 1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_x1, 'jac': wall_x1_jac})
            
            def wall_x2(params, i=i):
                return 1.0 - params[3*i] - params[3*i+2]
            def wall_x2_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i] = -1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_x2, 'jac': wall_x2_jac})
            
            def wall_y1(params, i=i):
                return params[3*i+1] - params[3*i+2]
            def wall_y1_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i+1] = 1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_y1, 'jac': wall_y1_jac})
            
            def wall_y2(params, i=i):
                return 1.0 - params[3*i+1] - params[3*i+2]
            def wall_y2_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i+1] = -1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_y2, 'jac': wall_y2_jac})
        
        iu = np.triu_indices(N, k=1)
        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            
            def pair_dist(params, i=i, j=j):
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - params[3*i+2] - params[3*j+2]
                
            def pair_dist_jac(params, i=i, j=j):
                g = np.zeros_like(params)
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dist = 1e-10
                g[3*i] = dx / dist
                g[3*j] = -dx / dist
                g[3*i+1] = dy / dist
                g[3*j+1] = -dy / dist
                g[3*i+2] = -1.0
                g[3*j+2] = -1.0
                return g
                
            cons.append({'type': 'ineq', 'fun': pair_dist, 'jac': pair_dist_jac})

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
                best_p.flatten(),
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res.success:
                p_new = res.x.reshape(N, 3)
                p_new = _enforce_validity(p_new)
                s_new = _score_valid(p_new)
                
                if s_new > best_score:
                    best_score = s_new
                    best_p = p_new
                    
        except Exception:
            pass
            
    return best_p


def _shake_and_refine(p, steps=5000, time_limit=20.0, seed=42):
    """
    Refine a valid configuration using random perturbations (shake and solve).
    This helps escape local minima found by gradient-based methods.
    """
    current_score = _score_valid(p)
    best_p = p.copy()
    best_score = current_score
    rng = np.random.RandomState(seed)
    
    start = time.time()
    sigma = 0.02
    
    for _ in range(steps):
        if time.time() - start > time_limit:
            break
            
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, sigma * 0.5, size=N)
        
        p_new = _enforce_validity(p_new)
        
        s_new = _score_valid(p_new)
        
        if s_new > best_score:
            best_score = s_new
            best_p = p_new
            
    return best_p


def _high_precision_polish(p, time_limit=10.0, seed=42):
    """
    High-precision final polish using SLSQP with very tight tolerances.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    start = time.time()
    
    x = best_p[:, 0]
    y = best_p[:, 1]
    r = best_p[:, 2]
    
    def obj(params):
        return -np.sum(params[2::3])
    
    def obj_grad(params):
        g = np.zeros_like(params)
        g[2::3] = -1.0
        return g
    
    cons = []
    
    for i in range(N):
        def wall_x1(params, i=i):
            return params[3*i] - params[3*i+2]
        def wall_x1_jac(params, i=i):
            g = np.zeros_like(params)
            g[3*i] = 1.0
            g[3*i+2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_x1, 'jac': wall_x1_jac})
        
        def wall_x2(params, i=i):
            return 1.0 - params[3*i] - params[3*i+2]
        def wall_x2_jac(params, i=i):
            g = np.zeros_like(params)
            g[3*i] = -1.0
            g[3*i+2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_x2, 'jac': wall_x2_jac})
        
        def wall_y1(params, i=i):
            return params[3*i+1] - params[3*i+2]
        def wall_y1_jac(params, i=i):
            g = np.zeros_like(params)
            g[3*i+1] = 1.0
            g[3*i+2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_y1, 'jac': wall_y1_jac})
        
        def wall_y2(params, i=i):
            return 1.0 - params[3*i+1] - params[3*i+2]
        def wall_y2_jac(params, i=i):
            g = np.zeros_like(params)
            g[3*i+1] = -1.0
            g[3*i+2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_y2, 'jac': wall_y2_jac})
    
    iu = np.triu_indices(N, k=1)
    for k in range(len(iu[0])):
        i = iu[0][k]
        j = iu[1][k]
        
        def pair_dist(params, i=i, j=j):
            dx = params[3*i] - params[3*j]
            dy = params[3*i+1] - params[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - params[3*i+2] - params[3*j+2]
            
        def pair_dist_jac(params, i=i, j=j):
            g = np.zeros_like(params)
            dx = params[3*i] - params[3*j]
            dy = params[3*i+1] - params[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-10:
                dist = 1e-10
            g[3*i] = dx / dist
            g[3*j] = -dx / dist
            g[3*i+1] = dy / dist
            g[3*j+1] = -dy / dist
            g[3*i+2] = -1.0
            g[3*j+2] = -1.0
            return g
            
        cons.append({'type': 'ineq', 'fun': pair_dist, 'jac': pair_dist_jac})

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
            best_p.flatten(),
            method='SLSQP',
            jac=obj_grad,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        
        if res.success:
            p_new = res.x.reshape(N, 3)
            p_new = _enforce_validity(p_new)
            s_new = _score_valid(p_new)
            
            if s_new > best_score:
                best_p = p_new
                best_score = s_new
                
    except Exception:
        pass
    
    return best_p


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_params = None
    best_score = -np.inf

    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        for init_fn in initializers:
            if time.time() > deadline - 15:
                break
            
            x0 = init_fn(seed)
            x0 = _enforce_validity(x0).flatten()
            
            time_per_pbd = 4.0
            if time.time() + time_per_pbd > deadline - 15:
                continue
                
            try:
                p_pbd = _pbd_refine(x0.reshape(N, 3), time_limit=time_per_pbd, seed=seed)
                s_pbd = _score_valid(p_pbd)
                
                if s_pbd > best_score:
                    best_score = s_pbd
                    best_params = p_pbd.copy()
                    
            except Exception:
                continue

    if best_params is None:
        best_params = _enforce_validity(_grid_initial(42)).reshape(N, 3)

    # Phase 2: SLSQP refinement
    remaining_for_slsqp = deadline - time.time() - 20.0
    if remaining_for_slsqp > 5.0:
        slsqp_seeds = [1, 2, 3, 4, 5, 6, 7, 8]
        for s_seed in slsqp_seeds:
            if time.time() > deadline - 20.0:
                break
            t_limit = min(5.0, (deadline - time.time() - 20.0) / max(1, len(slsqp_seeds) - s_seed + 1))
            if t_limit < 2.0:
                break
            try:
                p_slsqp = _slsqp_refine(best_params, time_limit=t_limit, seed=s_seed)
                s_slsqp = _score_valid(p_slsqp)
                if s_slsqp > best_score:
                    best_score = s_slsqp
                    best_params = p_slsqp
            except Exception:
                continue

    # Phase 3: Shake and refine
    remaining_for_shake = deadline - time.time() - 15.0
    if remaining_for_shake > 5.0:
        shake_seeds = [10, 11, 12, 13, 14]
        for sh_seed in shake_seeds:
            if time.time() > deadline - 15.0:
                break
            t_limit = min(5.0, (deadline - time.time() - 15.0) / max(1, len(shake_seeds) - shake_seeds.index(sh_seed)))
            if t_limit < 2.0:
                break
            try:
                p_shake = _shake_and_refine(best_params, steps=10000, time_limit=t_limit, seed=sh_seed)
                s_shake = _score_valid(p_shake)
                if s_shake > best_score:
                    best_score = s_shake
                    best_params = p_shake
            except Exception:
                continue

    # Phase 4: Final high-precision polish
    remaining_for_polish = deadline - time.time() - 3.0
    if remaining_for_polish > 3.0:
        try:
            p_polished = _high_precision_polish(best_params, time_limit=min(10.0, remaining_for_polish), seed=999)
            s_polished = _score_valid(p_polished)
            if s_polished > best_score:
                best_score = s_polished
                best_params = p_polished
        except Exception:
            pass

    # Final safety: enforce validity
    best_params = _enforce_validity(best_params)
    return best_params
