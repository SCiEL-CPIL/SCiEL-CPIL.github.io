"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 145.0


def _elapsed(start):
    return time.time() - start


def _remaining(start, deadline):
    return max(0.0, deadline - time.time())


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(p):
    """Check if configuration is valid (no overlaps, within bounds)."""
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return False
    if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < 1e-8):
        return False
    if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
        return False
    if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
        return False
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(r[iu[0]] + r[iu[1]] - dist > 1e-9):
        return False
    return True


def _enforce_validity(params):
    """Enforce all constraints by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-8, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-8))

    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-12
        if not np.any(active):
            break
        active_idx = np.where(active)[0]
        for k in active_idx:
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-8)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-8)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-8)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-8)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-8))

    return p


def _score(p):
    return np.sum(p[:, 2])


def _objective_grad(x, penalty=1000.0, smooth_eps=1e-5):
    """Objective: minimize -sum(r) + penalty * (pair_violations + wall_violations)."""
    p = x.reshape(N, 3)
    xi, yi, ri = p[:, 0], p[:, 1], np.maximum(p[:, 2], 1e-8)
    
    obj = -np.sum(ri)
    
    # Wall violations
    lx = ri - xi
    ux = xi + ri - 1.0
    ly = ri - yi
    uy = yi + ri - 1.0
    wall_pen = (np.sum(_softplus(lx, smooth_eps)**2) + np.sum(_softplus(ux, smooth_eps)**2) +
                np.sum(_softplus(ly, smooth_eps)**2) + np.sum(_softplus(uy, smooth_eps)**2))
    
    # Pair violations
    iu = np.triu_indices(N, k=1)
    dx = xi[iu[0]] - xi[iu[1]]
    dy = yi[iu[0]] - yi[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    ov = (ri[iu[0]] + ri[iu[1]]) - dist
    pair_pen = np.sum(_softplus(ov, smooth_eps)**2)
    
    obj += penalty * (pair_pen + wall_pen)
    
    # Gradient
    g = np.zeros_like(x)
    g[2::3] = -1.0
    
    def d_softplus_sq(z, eps):
        sp = _softplus(z, eps)
        sig = _sigmoid(z, eps)
        return 2.0 * sp * sig
    
    # Wall gradients
    for i in range(N):
        for z, xi_idx, ri_idx, sign_x, sign_r in [
            (lx[i], 3*i, 3*i+2, -1, 1),
            (ux[i], 3*i, 3*i+2, 1, 1),
            (ly[i], 3*i+1, 3*i+2, -1, 1),
            (uy[i], 3*i+1, 3*i+2, 1, 1),
        ]:
            dz = d_softplus_sq(z, smooth_eps)
            g[xi_idx] += penalty * sign_x * dz
            g[ri_idx] += penalty * sign_r * dz
    
    # Pair gradients
    dist_safe = np.maximum(dist, 1e-10)
    ux = dx / dist_safe
    uy = dy / dist_safe
    dz = d_softplus_sq(ov, smooth_eps)
    
    np.add.at(g[0::3], iu[0], -penalty * ux * dz)
    np.add.at(g[0::3], iu[1], penalty * ux * dz)
    np.add.at(g[1::3], iu[0], -penalty * uy * dz)
    np.add.at(g[1::3], iu[1], penalty * uy * dz)
    np.add.at(g[2::3], iu[0], penalty * dz)
    np.add.at(g[2::3], iu[1], penalty * dz)
    
    return obj, g


def _polish_analytic(params, deadline):
    """High-precision SLSQP polish on Delaunay contact edges."""
    p = params.reshape(N, 3).copy()
    centers = p[:, :2]
    
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
        edges = list(edge_set)
    except Exception:
        return params
    
    if not edges:
        return params
    
    eu = np.array([e[0] for e in edges])
    ev = np.array([e[1] for e in edges])
    m = len(eu)
    
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0; hi[3*i] = 1.0
        lo[3*i+1] = 0.0; hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-8; hi[3*i+2] = 0.5
    bounds_list = list(zip(lo, hi))
    
    def obj(x):
        return -np.sum(x[2::3])
    
    def obj_grad(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g
    
    def pair_con(x):
        xi = x[eu * 3]; yi = x[eu * 3 + 1]; ri = np.maximum(x[eu * 3 + 2], 1e-8)
        xj = x[ev * 3]; yj = x[ev * 3 + 1]; rj = np.maximum(x[ev * 3 + 2], 1e-8)
        dx = xi - xj; dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - (ri + rj)
    
    def pair_jac(x):
        J = np.zeros((m, 3 * N))
        xi = x[eu * 3]; yi = x[eu * 3 + 1]
        xj = x[ev * 3]; yj = x[ev * 3 + 1]
        dx = xi - xj; dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        ds = np.maximum(dist, 1e-10)
        ux = dx / ds; uy = dy / ds
        for k in range(m):
            i = eu[k]; j = ev[k]
            J[k, 3*i] = ux[k]; J[k, 3*i+1] = uy[k]; J[k, 3*i+2] = -1.0
            J[k, 3*j] = -ux[k]; J[k, 3*j+1] = -uy[k]; J[k, 3*j+2] = -1.0
        return J
    
    def wall_con(x):
        xi = x[0::3]; yi = x[1::3]; ri = np.maximum(x[2::3], 1e-8)
        return np.minimum(np.minimum(xi - ri, 1.0 - xi - ri),
                          np.minimum(yi - ri, 1.0 - yi - ri))
    
    def wall_jac(x):
        J = np.zeros((N, 3 * N))
        xi = x[0::3]; yi = x[1::3]; ri = np.maximum(x[2::3], 1e-8)
        for i in range(N):
            wmin = min(xi[i]-ri[i], 1-xi[i]-ri[i], yi[i]-ri[i], 1-yi[i]-ri[i])
            if xi[i]-ri[i] <= wmin + 1e-12: J[i, 3*i] += 1; J[i, 3*i+2] -= 1
            if 1-xi[i]-ri[i] <= wmin + 1e-12: J[i, 3*i] -= 1; J[i, 3*i+2] -= 1
            if yi[i]-ri[i] <= wmin + 1e-12: J[i, 3*i+1] += 1; J[i, 3*i+2] -= 1
            if 1-yi[i]-ri[i] <= wmin + 1e-12: J[i, 3*i+1] -= 1; J[i, 3*i+2] -= 1
        return J
    
    constraints = [
        {'type': 'ineq', 'fun': pair_con, 'jac': pair_jac},
        {'type': 'ineq', 'fun': wall_con, 'jac': wall_jac}
    ]
    
    try:
        res = minimize(obj, params, jac=obj_grad, method='SLSQP',
                       bounds=bounds_list, constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-15, 'maxfun': 800})
        if hasattr(res, 'x'):
            res_valid = _enforce_validity(res.x)
            if _score(res_valid) >= _score(params.reshape(N, 3)) - 1e-12:
                return res.x
    except Exception:
        pass
    
    return params


def _physics_relax(params, deadline, n_steps=200, dt=0.01):
    """Physics-based relaxation: repulsive forces for overlaps, attractive for growth."""
    p = params.reshape(N, 3).copy()
    best_p = p.copy()
    best_score = _score(p)
    start = time.time()
    
    for step in range(n_steps):
        if time.time() > deadline - 3:
            break
        
        forces = np.zeros_like(p)
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        ux = dx / dist_safe
        uy = dy / dist_safe
        
        # Repulsive force for overlaps
        ov = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = ov > 0
        rep = np.zeros(N)
        for k in np.where(active)[0]:
            i = iu[0][k]; j = iu[1][k]
            f = 50.0 * ov[k]
            forces[i, 0] += f * ux[k]
            forces[i, 1] += f * uy[k]
            forces[j, 0] -= f * ux[k]
            forces[j, 1] -= f * uy[k]
            # Grow radii
            forces[i, 2] += 0.5
            forces[j, 2] += 0.5
        
        # Wall forces
        for i in range(N):
            if p[i, 0] < p[i, 2]:
                forces[i, 0] += 100.0 * (p[i, 2] - p[i, 0])
            if p[i, 0] > 1 - p[i, 2]:
                forces[i, 0] -= 100.0 * (p[i, 0] + p[i, 2] - 1)
            if p[i, 1] < p[i, 2]:
                forces[i, 1] += 100.0 * (p[i, 2] - p[i, 1])
            if p[i, 1] > 1 - p[i, 2]:
                forces[i, 1] -= 100.0 * (p[i, 1] + p[i, 2] - 1)
        
        # Update
        p += dt * forces
        p[:, 0] = np.clip(p[:, 0], 0, 1)
        p[:, 1] = np.clip(p[:, 1], 0, 1)
        p[:, 2] = np.clip(p[:, 2], 1e-8, 0.5)
        
        # Enforce validity and track best
        p_valid = _enforce_validity(p)
        s = _score(p_valid)
        if s > best_score:
            best_score = s
            best_p = p_valid.copy()
    
    return best_p.reshape(-1)


def _initial_config_grid(seed):
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


def _initial_config_hex(seed):
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


def _initial_config_voronoi(seed):
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    radii = np.zeros(N)
    for i, pt in enumerate(points):
        dists = np.linalg.norm(points - pt, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([pt[0], 1-pt[0], pt[1], 1-pt[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.9
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _mma_optimize(params, deadline, max_iter=30):
    """
    MMA-style optimization: at each iteration, build a trust-region quadratic
    subproblem approximating the smooth penalty objective, solve it, and update.
    """
    p = params.copy()
    best_p = p
    best_score = _score(_enforce_validity(p))
    
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0; hi[3*i] = 1.0
        lo[3*i+1] = 0.0; hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-8; hi[3*i+2] = 0.5
    bounds_list = list(zip(lo, hi))
    
    # Start with a moderate penalty, increase over iterations
    penalty = 500.0
    smooth_eps = 1e-4
    delta = 0.1  # trust region radius
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        try:
            f, g = _objective_grad(p, penalty=penalty, smooth_eps=smooth_eps)
            
            # Build Hessian approximation (diagonal from gradient magnitude)
            H = np.maximum(np.abs(g), 1e-6)
            
            # Solve trust-region subproblem: min g^T d + 0.5 d^T H d s.t. ||d||_inf <= delta
            # Use scipy's trust-ncg or just projected gradient step
            # Simpler: use L-BFGS-B with the smooth objective for one step
            
            res = minimize(lambda x: _objective_grad(x, penalty=penalty, smooth_eps=smooth_eps)[0],
                           p,
                           jac=lambda x: _objective_grad(x, penalty=penalty, smooth_eps=smooth_eps)[1],
                           method='L-BFGS-B',
                           bounds=bounds_list,
                           options={'maxiter': 30, 'maxfun': 100, 'ftol': 1e-12})
            
            p_new = res.x
            p_valid = _enforce_validity(p_new)
            s = _score(p_valid)
            
            if s > best_score:
                best_score = s
                best_p = p_valid.copy()
                p = p_valid.copy()
                # Shrink trust region on success
                delta = max(0.01, delta * 0.7)
                # Gradually increase penalty
                penalty *= 1.5
            else:
                # Expand trust region on failure
                delta = min(0.5, delta * 1.5)
        except Exception:
            break
    
    return best_p.reshape(-1)


def _shake_and_refine(p, steps=3000, time_limit=15.0, seed=42, deadline=None):
    """Refine using random perturbations."""
    current_score = _score(p)
    best_p = p.copy()
    best_score = current_score
    rng = np.random.RandomState(seed)
    start = time.time()
    sigma = 0.015
    
    for step in range(steps):
        if time.time() - start > time_limit:
            break
        if deadline is not None and time.time() > deadline - 2:
            break
        
        progress = step / steps
        current_sigma = sigma * (1.0 - 0.5 * progress)
        
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, current_sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, current_sigma * 0.5, size=N)
        
        p_new = _enforce_validity(p_new)
        new_score = _score(p_new)
        
        if new_score > best_score + 1e-10:
            best_score = new_score
            best_p = p_new.copy()
    
    return best_p.reshape(-1)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0; hi[3*i] = 1.0
        lo[3*i+1] = 0.0; hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-8; hi[3*i+2] = 0.5
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    initializers = [
        lambda s: _initial_config_grid(s),
        lambda s: _initial_config_hex(s),
        lambda s: _initial_config_voronoi(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100, 13, 17, 19, 23]
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        for init_fn in initializers:
            if time.time() > deadline - 25:
                break
            
            params = init_fn(seed)
            params_valid = _enforce_validity(params)
            
            # Phase 1: MMA-style optimization
            if time.time() < deadline - 15:
                optimized = _mma_optimize(params_valid.reshape(-1), deadline)
                optimized_valid = _enforce_validity(optimized)
                opt_score = _score(optimized_valid)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_params = optimized_valid.copy()
            
            if best_params is None:
                continue
            
            # Phase 2: Physics relaxation
            if time.time() < deadline - 12:
                relaxed = _physics_relax(best_params, deadline, n_steps=150, dt=0.008)
                relaxed_valid = _enforce_validity(relaxed)
                relax_score = _score(relaxed_valid)
                
                if relax_score > best_score:
                    best_score = relax_score
                    best_params = relaxed_valid.copy()
            
            # Phase 3: Shake and refine
            if time.time() < deadline - 8:
                shaken = _shake_and_refine(best_params, steps=3000, time_limit=10.0,
                                           seed=seed, deadline=deadline)
                shaken_valid = _enforce_validity(shaken)
                shake_score = _score(shaken_valid)
                
                if shake_score > best_score:
                    best_score = shake_score
                    best_params = shaken_valid.copy()
            
            # Phase 4: Analytic SLSQP polish
            if time.time() < deadline - 5 and best_params is not None:
                polished = _polish_analytic(best_params, deadline)
                polished_valid = _enforce_validity(polished)
                polish_score = _score(polished_valid)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_params = polished_valid.copy()

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            final = _polish_analytic(best_params, deadline)
            final_valid = _enforce_validity(final)
            final_score = _score(final_valid)
            if final_score > best_score:
                best_score = final_score
                best_params = final_valid.copy()
        except Exception:
            pass

    return best_params
