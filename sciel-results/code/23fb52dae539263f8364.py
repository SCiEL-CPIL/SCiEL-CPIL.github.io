import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-10
DEADLINE = 145.0  # safety margin before 150s kill


def _penalty_objective(params, penalty=1000.0, smoothness=1e-4):
    """Smooth penalty objective using soft-min for overlaps."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall boundary violations (smooth)
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations (smooth with log-sum-exp style)
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy + EPS)
    overlap = (r[iu[0]] + r[iu[1]]) - dist
    # Smooth max(0, overlap) using softplus
    pair = np.sum(smoothness * np.log1p(np.exp(overlap / smoothness)))

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0, smoothness=1e-4):
    """Analytic gradient of smooth penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
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

    # Pairwise - smooth gradient
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy + EPS)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    # softplus derivative: sigmoid(ov/smoothness)
    sig = 1.0 / (1.0 + np.exp(-ov / smoothness))
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
    for _ in range(200):
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
        if max_ov < 1e-8:
            break

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


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


def _random_initial(seed):
    """Simple random initialization with radius estimation."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
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


def _corner_weighted_initial(seed):
    """Initialize with bias toward corners and edges for better packing."""
    rng = np.random.RandomState(seed)
    points = np.zeros((N, 2))
    
    # Place some circles near corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i in range(4):
        points[i] = corners[i] + rng.uniform(-0.05, 0.05, size=2)
    
    # Place some along edges
    for i in range(4, 12):
        if i % 4 == 0:
            points[i] = [rng.uniform(0.1, 0.9), 0.1]
        elif i % 4 == 1:
            points[i] = [rng.uniform(0.1, 0.9), 0.9]
        elif i % 4 == 2:
            points[i] = [0.1, rng.uniform(0.1, 0.9)]
        else:
            points[i] = [0.9, rng.uniform(0.1, 0.9)]
    
    # Remaining in center region
    for i in range(12, N):
        points[i] = rng.uniform(0.2, 0.8, size=2)
    
    # Estimate radii based on local density
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


def _lbfgs_optimize(p, penalty=1000.0, maxiter=1000, time_limit=5.0):
    """L-BFGS-B with smooth penalty, adaptive penalty annealing."""
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    
    # Adaptive penalty: start high, decrease as we converge
    penalties = [5000.0, 2000.0, 1000.0, 500.0, 200.0, 100.0, 50.0]
    smoothnesses = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    
    current = p.flatten().copy()
    
    for idx, (pen, sm) in enumerate(zip(penalties, smoothnesses)):
        if time.time() - start_time > time_limit - 0.5:
            break
        
        def obj(params):
            return _penalty_objective(params, penalty=pen, smoothness=sm)
        
        def grad(params):
            return _penalty_grad(params, penalty=pen, smoothness=sm)
        
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


def _slsqp_constrained(params):
    """Build SLSQP constraints."""
    p = params.reshape(N, 3)
    
    cons = []
    
    # Wall constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1.0 - p[3*i]) - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1.0 - p[3*i+1]) - p[3*i+2]})
    
    # Pairwise non-overlap
    iu = np.triu_indices(N, k=1)
    for k in range(len(iu[0])):
        i = iu[0][k]
        j = iu[1][k]
        def pair_fun(p, i=i, j=j):
            dx = p[3*i] - p[3*j]
            dy = p[3*i+1] - p[3*j+1]
            return np.sqrt(dx*dx + dy*dy + 1e-12) - p[3*i+2] - p[3*j+2]
        cons.append({'type': 'ineq', 'fun': pair_fun})
    
    return cons


def _slsqp_optimize(p, time_limit=10.0):
    """SLSQP with explicit constraints for high-precision polish."""
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    cons = _slsqp_constrained(p.flatten())
    
    def obj(params):
        pp = params.reshape(N, 3)
        return -np.sum(pp[:, 2])
    
    def obj_grad(params):
        g = np.zeros(3*N)
        g[2::3] = -1.0
        return g
    
    try:
        res = minimize(
            obj,
            p.flatten(),
            method='SLSQP',
            jac=obj_grad,
            bounds=bounds_list,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < -best_score + 1e-6:
            new_p = _enforce_validity(res.x.reshape(N, 3))
            new_score = _score_valid(new_p)
            if new_score > best_score:
                best_score = new_score
                best_p = new_p
    except Exception:
        pass
    
    return best_p, best_score


def _simulated_annealing(p, time_limit=15.0, seed=42):
    """Simulated annealing to escape local minima."""
    rng = np.random.RandomState(seed)
    start = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    current_p = p.copy()
    current_score = best_score
    
    T = 0.05  # initial temperature
    cooling = 0.9995
    
    while time.time() - start < time_limit:
        T *= cooling
        if T < 1e-6:
            break
        
        # Perturb
        p_new = current_p.copy()
        p_new[:, :2] += rng.normal(0, 0.01, size=(N, 2))
        p_new[:, 2] += rng.normal(0, 0.005, size=N)
        p_new = _enforce_validity(p_new)
        s_new = _score_valid(p_new)
        
        delta = s_new - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_p = p_new
            current_score = s_new
            if s_new > best_score:
                best_score = s_new
                best_p = p_new.copy()
        
        # Occasionally run L-BFGS polish
        if rng.random() < 0.05 and time.time() - start < time_limit - 5:
            p_polish, s_polish = _lbfgs_optimize(current_p, penalty=500.0, maxiter=200, time_limit=3.0)
            if s_polish > best_score:
                best_score = s_polish
                best_p = p_polish.copy()
            current_p = p_polish
            current_score = s_polish
    
    return best_p, best_score


def _high_precision_polish(p, time_limit=10.0):
    """Final high-precision polish using SLSQP with tight tolerances."""
    start = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    # Multiple SLSQP passes with decreasing perturbation
    for scale in [0.01, 0.005, 0.001, 0.0005]:
        if time.time() - start > time_limit - 2:
            break
        
        # Small perturbation to escape exact local minimum
        p_try = best_p.copy()
        rng = np.random.RandomState(12345)
        p_try[:, :2] += rng.normal(0, scale, size=(N, 2))
        p_try = _enforce_validity(p_try)
        
        p_polish, s_polish = _slsqp_optimize(p_try, time_limit=min(5.0, time_limit - (time.time() - start) - 1))
        if s_polish > best_score:
            best_score = s_polish
            best_p = p_polish.copy()
    
    return best_p, best_score


def _delaunay_polish(p, time_limit=8.0):
    """Use Delaunay triangulation to identify contact pairs and polish."""
    start = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    for iteration in range(5):
        if time.time() - start > time_limit - 2:
            break
        
        # Get Delaunay edges
        points = best_p[:, :2]
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
        except Exception:
            edges = set()
        
        # For each contact edge, try to push circles apart slightly and increase radii
        p_try = best_p.copy()
        for (i, j) in edges:
            dx = p_try[j, 0] - p_try[i, 0]
            dy = p_try[j, 1] - p_try[i, 1]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            if dist < 1e-10:
                continue
            # Push apart by tiny amount
            push = 0.001
            p_try[i, 0] -= push * dx / dist
            p_try[i, 1] -= push * dy / dist
            p_try[j, 0] += push * dx / dist
            p_try[j, 1] += push * dy / dist
        
        p_try = _enforce_validity(p_try)
        s_try = _score_valid(p_try)
        
        if s_try > best_score:
            best_score = s_try
            best_p = p_try.copy()
        
        # L-BFGS polish
        if time.time() - start < time_limit - 3:
            p_polish, s_polish = _lbfgs_optimize(best_p, penalty=200.0, maxiter=100, time_limit=3.0)
            if s_polish > best_score:
                best_score = s_polish
                best_p = p_polish.copy()
    
    return best_p, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        _random_initial,
        _grid_initial,
        _hex_initial,
        _corner_weighted_initial,
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100,
             1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]
    
    # Phase 1: L-BFGS with adaptive penalty
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        for init_fn in initializers:
            if time.time() > deadline - 20:
                break
            
            x0 = init_fn(seed)
            p0 = _enforce_validity(x0)
            
            try:
                p, s = _lbfgs_optimize(p0, penalty=1000.0, maxiter=500, time_limit=min(4.0, max(0.5, deadline - time.time() - 25)))
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
            except Exception:
                continue

    if best_params is None:
        best_params = _enforce_validity(_grid_initial(42))
        best_score = _score_valid(best_params)

    # Phase 2: Simulated annealing to escape local minima
    if time.time() < deadline - 15:
        try:
            p_sa, s_sa = _simulated_annealing(
                best_params,
                time_limit=min(15.0, max(1.0, deadline - time.time() - 15)),
                seed=42
            )
            if s_sa > best_score:
                best_score = s_sa
                best_params = p_sa
        except Exception:
            pass

    # Phase 3: Delaunay-based polish
    if time.time() < deadline - 10:
        try:
            p_del, s_del = _delaunay_polish(
                best_params,
                time_limit=min(8.0, max(1.0, deadline - time.time() - 10))
            )
            if s_del > best_score:
                best_score = s_del
                best_params = p_del
        except Exception:
            pass

    # Phase 4: High-precision SLSQP polish
    if time.time() < deadline - 5:
        try:
            p_final, s_final = _high_precision_polish(
                best_params,
                time_limit=min(10.0, max(1.0, deadline - time.time() - 5))
            )
            if s_final > best_score:
                best_score = s_final
                best_params = p_final
        except Exception:
            pass

    # Phase 5: Final tiny radius increases
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        
        p = best_params.copy()
        p[:, 2] *= 1.0005
        p_new = _enforce_validity(p)
        s_new = _score_valid(p_new)
        
        if s_new > best_score:
            best_score = s_new
            best_params = p_new
        else:
            break

    return best_params.copy()
