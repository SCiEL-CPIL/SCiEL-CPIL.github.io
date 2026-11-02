"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz, tol=1e-7):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
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

def _physics_relax(xyz, time_limit=4.0, grow_rate=0.001):
    """
    Force-directed physics relaxation that also grows radii over time.
    Primary optimization engine - handles packing constraints via repulsion.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    best_score = np.sum(r)
    best_xyz = xyz.copy()

    dt = 0.005

    for _ in range(8000):
        if time.time() - start_time > time_limit:
            break

        # Grow radii slowly to push packing tighter
        r *= (1.0 + grow_rate)

        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

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
                    force = (min_dist - dist) * 50.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 100.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 100.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 100.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 100.0

        x += dt * fx
        y += dt * fy

        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])

        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r

        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()

    return best_xyz

def _enforce_validity(params):
    """Enforce all constraints by shrinking radii and clipping positions."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

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

def _objective_full(params, penalty_weight=1000.0):
    """Full pairwise objective for high-precision polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    pen_x_lo = np.maximum(0, -(x - r))
    pen_x_hi = np.maximum(0, x + r - 1)
    pen_y_lo = np.maximum(0, -(y - r))
    pen_y_hi = np.maximum(0, y + r - 1)
    pen_r = np.maximum(0, -r)
    penalty = (np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
               + np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
               + np.sum(pen_r ** 2))

    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2

    return -np.sum(r) + penalty_weight * penalty

def _objective_delaunay(params, edges_cache, penalty_weight=50.0):
    """
    Objective using Delaunay-based constraints (cheap) for the main search.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = (np.sum(np.maximum(0, -(x - r)) ** 2)
               + np.sum(np.maximum(0, x + r - 1) ** 2)
               + np.sum(np.maximum(0, -(y - r)) ** 2)
               + np.sum(np.maximum(0, y + r - 1) ** 2)
               + np.sum(np.maximum(0, -r) ** 2))

    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edges_cache['edges'] = edges
    edge_set = set(edges)

    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap ** 2

    # Safety: check a few close pairs not in Delaunay
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edge_set or (j, i) in edge_set:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            max_r = max(r[i], r[j])
            if dist_sq < (4 * max_r * max_r):
                dist = np.sqrt(dist_sq)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2

    return -np.sum(r) + penalty_weight * penalty

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
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
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_config_force_directed(seed):
    """Generate initial config using a simple force-directed relaxation."""
    np.random.seed(seed)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.full(N, 0.1)

    for _ in range(100):
        moved = False
        for i in range(N):
            fx, fy = 0, 0
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] and dist > 1e-6:
                    f = (r[i] + r[j] - dist) / dist
                    fx += f * dx
                    fy += f * dy
            if x[i] < r[i]:
                fx += (r[i] - x[i]) * 10
            if x[i] > 1 - r[i]:
                fx -= (x[i] - (1 - r[i])) * 10
            if y[i] < r[i]:
                fy += (r[i] - y[i]) * 10
            if y[i] > 1 - r[i]:
                fy -= (y[i] - (1 - r[i])) * 10

            x[i] += 0.01 * fx
            y[i] += 0.01 * fy
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            if np.abs(fx) > 1e-4 or np.abs(fy) > 1e-4:
                moved = True
        if not moved:
            break

    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])

    return np.column_stack([x, y, r])

def _optimize_delaunay(xyz, time_budget=30.0):
    """
    Optimize using Delaunay-based constraints, then polish with full pairwise.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()

    edges_cache = {'edges': None}

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    # Phase 1: Delaunay-based optimization
    try:
        result = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(edges_cache,),
                          options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_delaunay(best_params, edges_cache):
            best_params = result.x
    except Exception:
        pass

    if time.time() - start_time > time_budget:
        return best_params.reshape(N, 3)

    # Phase 2: Polish with full pairwise constraints
    try:
        result = minimize(_objective_full, best_params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_full(best_params):
            best_params = result.x
    except Exception:
        pass

    return best_params.reshape(N, 3)

def _polish_full(xyz, time_budget=5.0):
    """
    Final high-precision polish with escalating penalty weights and tight
    tolerances to remove any residual overlap and squeeze out last radii.
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
                best_params = result.x
                best_obj = result.fun
        except Exception:
            pass

    return best_params.reshape(N, 3)

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])

    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_initial_config_grid())
    initial_configs.append(_initial_config_hex())

    for seed in [1, 2, 3, 4, 5]:
        initial_configs.append(_initial_config_force_directed(seed))

    np.random.seed(42)
    for _trial in range(3):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))

    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 8.0:
            break

        time_budget = min(20.0, remaining - 3.0)
        if time_budget < 3.0:
            break

        # Physics relaxation to strengthen the starting point
        xyz_phys = _physics_relax(xyz0, time_limit=min(4.0, time_budget * 0.4))

        # Delaunay search + full polish
        xyz_opt = _optimize_delaunay(xyz_phys, time_budget=time_budget)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt

    # Final high-precision polish on the best configuration
    elapsed = time.time() - global_start
    if elapsed < deadline - 6.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 2.0:
            # Jitter and re-optimize to escape local minima
            np.random.seed(123)
            jittered = best_xyz + np.random.normal(0, 0.01, best_xyz.shape)
            for i in range(N):
                jittered[i, 0] = np.clip(jittered[i, 0], 0, 1)
                jittered[i, 1] = np.clip(jittered[i, 1], 0, 1)
                jittered[i, 2] = np.clip(jittered[i, 2], 0.01, 0.5)

            xyz_polished = _optimize_delaunay(jittered, time_budget=remaining * 0.5)
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_polished

    # Escalating-penalty precision polish on the best configuration
    elapsed = time.time() - global_start
    if elapsed < deadline - 3.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 1.0:
            xyz_polished = _polish_full(best_xyz, time_budget=remaining)
            if _validate(xyz_polished):
                score = np.sum(xyz_polished[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_polished

    # Final check
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz.flatten())
        if not _validate(best_xyz):
            best_xyz = _fix_config(best_xyz)

    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()

    return best_xyz

# EVOLVE-BLOCK-END
