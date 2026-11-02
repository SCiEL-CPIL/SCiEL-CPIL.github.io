"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_constraints(p, iu):
    """Return pairwise distance - (r_i + r_j) for all pairs."""
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist - (r[iu[0]] + r[iu[1]])


def _penalty_objective(params, penalty=1000.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0):
    """Analytic gradient of penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions
    for i in range(n):
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

    # Pairwise
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            g[3*i] += penalty * (-adx[k])
            g[3*j] += penalty * (+adx[k])
            g[3*i+1] += penalty * (-ady[k])
            g[3*j+1] += penalty * (+ady[k])
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _voronoi_initial(seed):
    """Initialize using random points with estimated radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.9
        
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _hex_initial(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    # Generate a 7x7 hex grid
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
    
    # Select N points closest to center
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
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    for _ in range(1000):
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
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _pbd_simulate(params, max_iter=5000, dt=0.02, pressure=0.5,
                  stiffness=0.5, deadline=None):
    """
    Physics-based Position Based Dynamics simulation.
    Circles are particles with mass proportional to r^2.
    Subject to outward pressure and collision constraints.
    Uses Gauss-Seidel relaxation for distance and boundary constraints.
    """
    p = params.reshape(N, 3).copy()
    x = p[:, 0].copy()
    y = p[:, 1].copy()
    r = np.maximum(p[:, 2], 1e-6).copy()
    
    # Mass proportional to r^2
    mass = np.maximum(r * r, 1e-6)
    inv_mass = 1.0 / mass
    
    # Velocity
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    # Precompute pair indices
    iu = np.triu_indices(N, k=1)
    
    best_score = _score_valid(p)
    best_p = p.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # --- Apply pressure force (outward from center of mass) ---
        # This encourages circles to expand and push against each other/walls
        cx = np.mean(x)
        cy = np.mean(y)
        for i in range(N):
            dx = x[i] - cx
            dy = y[i] - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-6:
                fx = pressure * dx / dist
                fy = pressure * dy / dist
            else:
                fx = 0.0
                fy = 0.0
            vx[i] += fx * dt * inv_mass[i]
            vy[i] += fy * dt * inv_mass[i]
        
        # --- Velocity damping ---
        vx *= 0.9
        vy *= 0.9
        
        # --- Predict positions ---
        px = x + vx * dt
        py = y + vy * dt
        
        # --- Gauss-Seidel constraint relaxation ---
        # Iterate multiple times for convergence
        for _ in range(10):
            # Boundary constraints
            for i in range(N):
                # Left wall
                if px[i] < r[i]:
                    px[i] = r[i]
                    vx[i] = 0.0
                # Right wall
                if px[i] > 1.0 - r[i]:
                    px[i] = 1.0 - r[i]
                    vx[i] = 0.0
                # Bottom wall
                if py[i] < r[i]:
                    py[i] = r[i]
                    vy[i] = 0.0
                # Top wall
                if py[i] > 1.0 - r[i]:
                    py[i] = 1.0 - r[i]
                    vy[i] = 0.0
            
            # Pairwise collision constraints
            for idx in range(len(iu[0])):
                i = iu[0][idx]
                j = iu[1][idx]
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    if dist < 1e-10:
                        # Circles at same position, push apart
                        dx = 1.0
                        dy = 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = min_dist - dist
                    # Weighted correction based on inverse mass
                    w_i = inv_mass[i] / (inv_mass[i] + inv_mass[j])
                    w_j = inv_mass[j] / (inv_mass[i] + inv_mass[j])
                    
                    px[i] += overlap * dx * w_i
                    py[i] += overlap * dy * w_i
                    px[j] -= overlap * dx * w_j
                    py[j] -= overlap * dy * w_j
                    
                    # Damp velocity along collision normal
                    rel_vx = vx[i] - vx[j]
                    rel_vy = vy[i] - vy[j]
                    vel_along = rel_vx * dx + rel_vy * dy
                    if vel_along < 0:
                        vx[i] -= vel_along * dx * w_i
                        vy[i] -= vel_along * dy * w_i
                        vx[j] += vel_along * dx * w_j
                        vy[j] += vel_along * dy * w_j
        
        x = px
        y = py
        
        # Update radii: try to grow circles
        # Each circle can grow up to its minimum distance to walls and other circles
        new_r = np.zeros(N)
        for i in range(N):
            wall_dist = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            min_neighbor = wall_dist
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_neighbor = min(min_neighbor, (dist - r[j]))
            # Grow towards the limiting distance
            target_r = max(min_neighbor, 1e-6)
            # Smooth growth
            new_r[i] = r[i] + 0.1 * (target_r - r[i])
            new_r[i] = max(new_r[i], 1e-6)
            new_r[i] = min(new_r[i], 0.5)
        r = new_r
        
        p[:, 0] = x
        p[:, 1] = y
        p[:, 2] = r
        
        # Check validity and score
        p_valid = _enforce_validity(p)
        score = _score_valid(p_valid)
        if score > best_score:
            best_score = score
            best_p = p_valid.copy()
        
        # Early termination if converged
        if it > 100 and score < best_score - 1e-8:
            break
    
    return best_p.reshape(-1)


def _grow_relax_step(params, bounds_list, deadline):
    """
    One grow-and-relax iteration using SLSQP on Delaunay edges.
    """
    p = params.reshape(N, 3).copy()
    centers = p[:, :2]
    radii = np.maximum(p[:, 2], 1e-6)
    
    # Build Delaunay edges
    try:
        tri = Delaunay(centers)
        edge_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edge_set.add((a, b))
    except Exception:
        edge_set = set()
    
    if len(edge_set) == 0:
        return params
    
    edge_list = list(edge_set)
    eu = np.array([e[0] for e in edge_list])
    ev = np.array([e[1] for e in edge_list])
    m = len(eu)
    
    def obj(x):
        return -np.sum(x[2::3])
    
    def obj_grad(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g
    
    def constraint_func(x):
        xi = x[eu * 3]
        yi = x[eu * 3 + 1]
        ri = np.maximum(x[eu * 3 + 2], 1e-6)
        xj = x[ev * 3]
        yj = x[ev * 3 + 1]
        rj = np.maximum(x[ev * 3 + 2], 1e-6)
        dx = xi - xj
        dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - (ri + rj)
    
    def constraint_grad(x):
        jacobian = np.zeros((m, 3 * N))
        xi = x[eu * 3]
        yi = x[eu * 3 + 1]
        xj = x[ev * 3]
        yj = x[ev * 3 + 1]
        dx = xi - xj
        dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        ux = dx / dist_safe
        uy = dy / dist_safe
        
        for k in range(m):
            i = eu[k]
            j = ev[k]
            jacobian[k, 3*i] = ux[k]
            jacobian[k, 3*i+1] = uy[k]
            jacobian[k, 3*i+2] = -1.0
            jacobian[k, 3*j] = -ux[k]
            jacobian[k, 3*j+1] = -uy[k]
            jacobian[k, 3*j+2] = -1.0
        return jacobian
    
    def wall_func(x):
        xi = x[0::3]
        yi = x[1::3]
        ri = np.maximum(x[2::3], 1e-6)
        return np.minimum(np.minimum(xi - ri, 1.0 - xi - ri),
                          np.minimum(yi - ri, 1.0 - yi - ri))
    
    def wall_grad(x):
        jacobian = np.zeros((N, 3 * N))
        xi = x[0::3]
        yi = x[1::3]
        ri = np.maximum(x[2::3], 1e-6)
        
        for i in range(N):
            wx_left = xi[i] - ri[i]
            wx_right = 1.0 - xi[i] - ri[i]
            wy_bottom = yi[i] - ri[i]
            wy_top = 1.0 - yi[i] - ri[i]
            wmin = min(wx_left, wx_right, wy_bottom, wy_top)
            
            if wx_left <= wmin + 1e-12:
                jacobian[i, 3*i] += 1.0
                jacobian[i, 3*i+2] -= 1.0
            if wx_right <= wmin + 1e-12:
                jacobian[i, 3*i] -= 1.0
                jacobian[i, 3*i+2] -= 1.0
            if wy_bottom <= wmin + 1e-12:
                jacobian[i, 3*i+1] += 1.0
                jacobian[i, 3*i+2] -= 1.0
            if wy_top <= wmin + 1e-12:
                jacobian[i, 3*i+1] -= 1.0
                jacobian[i, 3*i+2] -= 1.0
        return jacobian
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_grad},
        {'type': 'ineq', 'fun': wall_func, 'jac': wall_grad}
    ]
    
    try:
        res = minimize(obj, params, jac=obj_grad, method='SLSQP',
                       bounds=bounds_list, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-12, 'maxfun': 500})
        
        if hasattr(res, 'x'):
            res_valid = _enforce_validity(res.x)
            if _score_valid(res_valid) > _score_valid(params.reshape(N, 3)):
                return res.x
    except Exception:
        pass
    
    return params


def _grow_relax_loop(params, bounds_list, deadline, max_iter=8):
    """
    Iterative grow-and-relax: repeatedly build contact graph and optimize.
    """
    p = params
    best_p = p
    best_score = _score_valid(p.reshape(N, 3))
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        p = _grow_relax_step(p, bounds_list, deadline)
        p_valid = _enforce_validity(p)
        s = _score_valid(p_valid)
        
        if s > best_score + 1e-10:
            best_score = s
            best_p = p_valid.copy()
        else:
            break
    
    return best_p.reshape(-1)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

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
        if time.time() > deadline - 20:
            break
        for init_fn in initializers:
            if time.time() > deadline - 15:
                break
            
            params = init_fn(seed)

            # Phase 1: Quick penalty-based L-BFGS-B to get a valid starting point
            for pen in [500.0, 5000.0]:
                if time.time() > deadline - 12:
                    break
                try:
                    res = minimize(lambda p: _penalty_objective(p, penalty=pen),
                                   params,
                                   jac=lambda p: _penalty_grad(p, penalty=pen),
                                   method='L-BFGS-B',
                                   bounds=bounds_list,
                                   options={'maxiter': 80, 'maxfun': 150,
                                            'ftol': 1e-10, 'gtol': 1e-6})
                    params = res.x
                except Exception:
                    pass

            # Ensure valid state
            params_valid = _enforce_validity(params)
            current_score = _score_valid(params_valid)
            if current_score > best_score:
                best_score = current_score
                best_params = params_valid.copy()

            # Phase 2: PBD simulation (structural change - physics-based approach)
            if time.time() < deadline - 10:
                pbd_result = _pbd_simulate(params_valid.reshape(-1),
                                           max_iter=3000, dt=0.02,
                                           pressure=0.3, stiffness=0.5,
                                           deadline=deadline - 5)
                pbd_valid = _enforce_validity(pbd_result)
                pbd_score = _score_valid(pbd_valid)
                
                if pbd_score > best_score:
                    best_score = pbd_score
                    best_params = pbd_valid.copy()

            # Phase 3: Grow-and-relax loop (SLSQP on Delaunay edges)
            if time.time() < deadline - 8 and best_params is not None:
                refined = _grow_relax_loop(best_params.reshape(-1), bounds_list, deadline, max_iter=5)
                refined_valid = _enforce_validity(refined)
                refined_score = _score_valid(refined_valid)
                
                if refined_score > best_score:
                    best_score = refined_score
                    best_params = refined_valid.copy()

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
