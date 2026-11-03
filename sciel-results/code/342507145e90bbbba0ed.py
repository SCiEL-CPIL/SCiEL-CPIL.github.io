"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed(start):
    return time.time() - start


def _remaining(start, deadline):
    return max(0.0, deadline - time.time())


def _pairwise_constraints(p, iu):
    """Return pairwise distance - (r_i + r_j) for all pairs."""
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist - (r[iu[0]] + r[iu[1]])


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _penalty_objective(params, penalty=1000.0, smooth_eps=1e-4):
    """Penalty objective with smooth constraints: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)

    score = -np.sum(r)

    # Wall boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    wall = np.sum(_softplus(lx, smooth_eps)**2) + np.sum(_softplus(ux, smooth_eps)**2) + \
           np.sum(_softplus(ly, smooth_eps)**2) + np.sum(_softplus(uy, smooth_eps)**2)

    # Pairwise overlap violations (smooth)
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = (r[iu[0]] + r[iu[1]]) - dist
    pair = np.sum(_softplus(overlap, smooth_eps)**2)

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0, smooth_eps=1e-4):
    """Analytic gradient of smooth penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Helper for smooth derivative of softplus(z)^2
    def d_softplus_sq(z, eps):
        # d/dz [softplus(z,eps)^2] = 2 * softplus(z,eps) * sigmoid(z/eps)
        sp = _softplus(z, eps)
        sig = 1.0 / (1.0 + np.exp(-z / eps))
        return 2.0 * sp * sig

    # Wall contributions (smooth)
    for i in range(n):
        # Left wall: z = r - x
        z_lx = r[i] - x[i]
        dz = d_softplus_sq(z_lx, smooth_eps)
        g[3*i] += penalty * (-dz)
        g[3*i+2] += penalty * (dz)

        # Right wall: z = x + r - 1
        z_ux = x[i] + r[i] - 1.0
        dz = d_softplus_sq(z_ux, smooth_eps)
        g[3*i] += penalty * (dz)
        g[3*i+2] += penalty * (dz)

        # Bottom wall: z = r - y
        z_ly = r[i] - y[i]
        dz = d_softplus_sq(z_ly, smooth_eps)
        g[3*i+1] += penalty * (-dz)
        g[3*i+2] += penalty * (dz)

        # Top wall: z = y + r - 1
        z_uy = y[i] + r[i] - 1.0
        dz = d_softplus_sq(z_uy, smooth_eps)
        g[3*i+1] += penalty * (dz)
        g[3*i+2] += penalty * (dz)

    # Pairwise
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    # Only compute gradients for active or near-active constraints to save time
    # Using smooth gradient for all, but it's cheap enough for N=26
    dz = d_softplus_sq(ov, smooth_eps)
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    # Accumulate into g
    # For each pair k: i = iu[0][k], j = iu[1][k]
    # d/dx_i (ov^2) = 2*ov*sig * (-ux) -> penalty * (-ux * dz)
    # d/dx_j (ov^2) = 2*ov*sig * (ux)  -> penalty * (ux * dz)
    # d/dr_i (ov^2) = 2*ov*sig * (1)   -> penalty * (dz)
    # d/dr_j (ov^2) = 2*ov*sig * (1)   -> penalty * (dz)
    
    np.add.at(g[0::3], iu[0], -penalty * ux * dz)
    np.add.at(g[0::3], iu[1], penalty * ux * dz)
    np.add.at(g[1::3], iu[0], -penalty * uy * dz)
    np.add.at(g[1::3], iu[1], penalty * uy * dz)
    np.add.at(g[2::3], iu[0], penalty * dz)
    np.add.at(g[2::3], iu[1], penalty * dz)

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

    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-12
        if not np.any(active):
            break
        # Vectorized shrink: for each overlapping pair, shrink the smaller radius
        active_idx = np.where(active)[0]
        for k in active_idx:
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


def _shake_and_refine(p, steps=3000, time_limit=15.0, seed=42, deadline=None):
    """
    Refine a valid configuration using random perturbations (shake and solve).
    """
    current_score = _score_valid(p)
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
        
        # Adaptive sigma: decrease over time for finer search
        progress = step / steps
        current_sigma = sigma * (1.0 - 0.5 * progress)
        
        # Perturb positions and radii
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, current_sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, current_sigma * 0.5, size=N)
        
        # Enforce validity
        p_new = _enforce_validity(p_new)
        new_score = _score_valid(p_new)
        
        if new_score > best_score + 1e-10:
            best_score = new_score
            best_p = p_new.copy()
    
    return best_p.reshape(-1)


def _physics_relax(params, time_limit=10.0, deadline=None, start_time=None):
    """
    Physics-based relaxation: apply forces to push apart overlapping circles
    and grow radii where possible. Uses velocity-Verlet-like integration.
    """
    if start_time is None:
        start_time = time.time()
    
    p = params.reshape(N, 3).copy()
    v = np.zeros_like(p)
    best_p = p.copy()
    best_score = _score_valid(p)
    
    dt = 0.005
    damping = 0.95
    repulsion_strength = 50.0
    growth_strength = 1.0
    
    iu = np.triu_indices(N, k=1)
    
    while True:
        if time.time() - start_time > time_limit:
            break
        if deadline is not None and time.time() > deadline - 2:
            break
        
        # Compute forces
        f = np.zeros_like(p)
        
        # Repulsive forces between overlapping circles
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        
        if np.any(active):
            ux = dx / dist_safe
            uy = dy / dist_safe
            force_mag = repulsion_strength * np.maximum(overlap, 0)
            
            np.add.at(f[:, 0], iu[0], force_mag * ux)
            np.add.at(f[:, 0], iu[1], -force_mag * ux)
            np.add.at(f[:, 1], iu[0], force_mag * uy)
            np.add.at(f[:, 1], iu[1], -force_mag * uy)
            # Shrink radii of overlapping circles
            np.add.at(f[:, 2], iu[0], -force_mag * 0.5)
            np.add.at(f[:, 2], iu[1], -force_mag * 0.5)
        
        # Growth force: try to increase radii
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        # Only grow if there's room
        can_grow = (wall - p[:, 2]) > 1e-8
        f[:, 2] += growth_strength * can_grow.astype(float)
        
        # Wall repulsion
        for i in range(N):
            if p[i, 0] < p[i, 2]:
                f[i, 0] += repulsion_strength * (p[i, 2] - p[i, 0])
            if p[i, 0] > 1.0 - p[i, 2]:
                f[i, 0] -= repulsion_strength * (p[i, 0] - (1.0 - p[i, 2]))
            if p[i, 1] < p[i, 2]:
                f[i, 1] += repulsion_strength * (p[i, 2] - p[i, 1])
            if p[i, 1] > 1.0 - p[i, 2]:
                f[i, 1] -= repulsion_strength * (p[i, 1] - (1.0 - p[i, 2]))
        
        # Update velocities and positions
        v = v * damping + f * dt
        p += v * dt
        
        # Enforce bounds
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
        p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
        
        # Enforce validity and check score
        p_valid = _enforce_validity(p)
        score = _score_valid(p_valid)
        if score > best_score + 1e-12:
            best_score = score
            best_p = p_valid.copy()
            p = p_valid.copy()
    
    return best_p.reshape(-1)


def _polish_analytic(params, deadline=None, start_time=None):
    """
    High-precision polish using SLSQP with tight tolerances on the contact graph.
    """
    if start_time is None:
        start_time = time.time()
    
    p = params.reshape(N, 3).copy()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    
    # Build contact graph
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
        edge_list = list(edge_set)
    except Exception:
        edge_list = []
    
    if len(edge_list) == 0:
        return best_p.reshape(-1)
    
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
    
    # Multiple polish passes with decreasing tolerance
    for ftol in [1e-10, 1e-12, 1e-14]:
        if deadline is not None and time.time() > deadline - 3:
            break
        
        try:
            res = minimize(obj, p, jac=obj_grad, method='SLSQP',
                           bounds=bounds_list, constraints=constraints,
                           options={'maxiter': 300, 'ftol': ftol, 'maxfun': 1000})
            
            if hasattr(res, 'x'):
                res_valid = _enforce_validity(res.x)
                s = _score_valid(res_valid)
                if s > best_score + 1e-12:
                    best_score = s
                    best_p = res_valid.copy()
                    p = res_valid.copy()
        except Exception:
            pass
    
    return best_p.reshape(-1)


def _contact_edges(p):
    """Build contact graph edges using Delaunay triangulation."""
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
        return list(edge_set)
    except Exception:
        return []


def _grow_relax_step(params, bounds_list, deadline):
    """
    One grow-and-relax iteration using SLSQP on Delaunay edges.
    """
    p = params.reshape(N, 3).copy()
    centers = p[:, :2]
    radii = np.maximum(p[:, 2], 1e-6)
    
    # Build Delaunay edges
    edge_list = _contact_edges(centers)
    
    if len(edge_list) == 0:
        return params
    
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
                       options={'maxiter': 200, 'ftol': 1e-14, 'maxfun': 600})
        
        if hasattr(res, 'x'):
            res_valid = _enforce_validity(res.x)
            if _score_valid(res_valid) > _score_valid(params.reshape(N, 3)):
                return res.x
    except Exception:
        pass
    
    return params


def _grow_relax_loop(params, bounds_list, deadline, max_iter=6):
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


def _global_optimize(params, bounds_list, deadline, max_iter=50):
    """
    Global optimization using penalty method with L-BFGS-B and smooth constraints.
    """
    p = params.copy()
    best_p = p
    best_score = _score_valid(_enforce_validity(p))
    
    # Increased penalties and more iterations for better convergence
    penalties = [100.0, 500.0, 2000.0, 5000.0, 10000.0, 20000.0]
    
    for pen in penalties:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(lambda x: _penalty_objective(x, penalty=pen, smooth_eps=1e-5),
                           p,
                           jac=lambda x: _penalty_grad(x, penalty=pen, smooth_eps=1e-5),
                           method='L-BFGS-B',
                           bounds=bounds_list,
                           options={'maxiter': 150, 'maxfun': 300,
                                    'ftol': 1e-12, 'gtol': 1e-8})
            p = res.x
            p_valid = _enforce_validity(p)
            s = _score_valid(p_valid)
            if s > best_score:
                best_score = s
                best_p = p_valid.copy()
        except Exception:
            pass
    
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

    # Seeds for better exploration
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100, 13, 17, 19, 23]
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        for init_fn in initializers:
            if time.time() > deadline - 25:
                break
            
            params = init_fn(seed)

            # Phase 1: Physics-based relaxation
            if time.time() < deadline - 20:
                phys = _physics_relax(params, time_limit=8.0, deadline=deadline,
                                      start_time=start_time)
                phys_valid = _enforce_validity(phys)
                phys_score = _score_valid(phys_valid)
                
                if phys_score > best_score:
                    best_score = phys_score
                    best_params = phys_valid.copy()
            
            if best_params is None:
                continue

            # Phase 2: Global penalty-based optimization
            if time.time() < deadline - 15:
                optimized = _global_optimize(best_params, bounds_list, deadline)
                optimized_valid = _enforce_validity(optimized)
                opt_score = _score_valid(optimized_valid)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_params = optimized_valid.copy()

            # Phase 3: Shake and refine to escape local minima
            if time.time() < deadline - 10:
                shaken = _shake_and_refine(best_params, steps=3000, time_limit=8.0,
                                           seed=seed, deadline=deadline)
                shaken_valid = _enforce_validity(shaken)
                shake_score = _score_valid(shaken_valid)
                
                if shake_score > best_score:
                    best_score = shake_score
                    best_params = shaken_valid.copy()

            # Phase 4: Grow-and-relax loop (SLSQP on Delaunay edges)
            if time.time() < deadline - 5 and best_params is not None:
                refined = _grow_relax_loop(best_params, bounds_list, deadline, max_iter=5)
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

    # Final high-precision polish using analytic SLSQP
    if time.time() < deadline - 3:
        try:
            polished = _polish_analytic(best_params, deadline=deadline,
                                        start_time=start_time)
            polished_valid = _enforce_validity(polished)
            polished_score = _score_valid(polished_valid)
            if polished_score > best_score:
                best_score = polished_score
                best_params = polished_valid.copy()
        except Exception:
            pass

    return best_params
