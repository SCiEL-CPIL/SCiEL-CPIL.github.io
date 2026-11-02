import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-10
DEADLINE = 145.0  # safety margin before 150s kill


def _softplus(x):
    """Smooth approximation of max(0, x)."""
    # Avoid overflow in exp
    x = np.clip(x, -50, 50)
    return np.log(1.0 + np.exp(x))


def _get_delaunay_edges(points):
    """Get edges of the Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
        return list(edges)
    except Exception:
        # Fallback: all pairs (expensive but safe)
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _delaunay_objective(params, edges, penalty=1000.0, smoothness=1e-4):
    """
    Objective: minimize -sum(r) + penalty * (smooth overlaps + smooth wall violations).
    Overlaps are only checked for Delaunay neighbors to reduce cost.
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall boundary violations (smooth)
    wx = _softplus(r - x) + _softplus(r - (1.0 - x))
    wy = _softplus(r - y) + _softplus(r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations (smooth, only for Delaunay edges)
    pair_penalty = 0.0
    if edges:
        # Vectorized calculation for edges
        i_idx = np.array([e[0] for e in edges])
        j_idx = np.array([e[1] for e in edges])
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        
        # Constraint: dist >= r_i + r_j  =>  r_i + r_j - dist <= 0
        # Violation: max(0, r_i + r_j - dist)
        viol = _softplus(r[i_idx] + r[j_idx] - dist)
        pair_penalty = np.sum(viol)

    return score + penalty * (pair_penalty + wall)


def _delaunay_grad(params, edges, penalty=1000.0):
    """
    Gradient of the Delaunay objective.
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    g = np.zeros(3 * N)
    
    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall gradients
    # d(softplus(r-x))/dr = sigmoid(r-x), d/dx = -sigmoid(r-x)
    sx_l = 1 / (1 + np.exp(-(r - x)))
    sx_r = 1 / (1 + np.exp(-(r - (1 - x))))
    sy_l = 1 / (1 + np.exp(-(r - y)))
    sy_r = 1 / (1 + np.exp(-(r - (1 - y))))
    
    g[0::3] += penalty * (-sx_l + sx_r)
    g[1::3] += penalty * (-sy_l + sy_r)
    g[2::3] += penalty * (sx_l + sx_r + sy_l + sy_r)

    # Pairwise gradients
    if edges:
        i_idx = np.array([e[0] for e in edges])
        j_idx = np.array([e[1] for e in edges])
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        
        # softplus'(x) = sigmoid(x)
        # x = r_i + r_j - dist
        sig = 1 / (1 + np.exp(-(r[i_idx] + r[j_idx] - dist)))
        
        # d/dx_i of softplus(r_i + r_j - dist)
        # = sig * (1 - dx/dist)
        # d/dy_i of softplus(r_i + r_j - dist)
        # = sig * (1 - dy/dist)
        # d/dr_i of softplus(r_i + r_j - dist)
        # = sig * 1
        
        # Note: dist = sqrt(dx^2 + dy^2)
        # d(dist)/dx_i = -dx/dist
        # d(softplus)/dx_i = sig * (0 - d(dist)/dx_i) = sig * (dx/dist)
        # Wait. Let f = r_i + r_j - dist.
        # df/dx_i = - d(dist)/dx_i = - (-dx/dist) = dx/dist.
        # d(softplus(f))/dx_i = sigmoid(f) * df/dx_i = sig * dx/dist.
        
        # Let's re-verify signs.
        # If circles overlap, r_i + r_j > dist. f > 0. sig > 0.5.
        # We want to push them apart.
        # If x_i < x_j, dx = x_i - x_j < 0.
        # Gradient wrt x_i should be negative (push left).
        # sig * dx/dist is negative. Correct.
        
        gx = sig * dx / dist_safe
        gy = sig * dy / dist_safe
        gr = sig
        
        # Accumulate gradients
        # For each edge k, update i and j
        # g[3*i] += penalty * gx[k]
        # g[3*j] += penalty * (-gx[k])  (since dx = x_i - x_j, d/dx_j = -gx)
        
        np.add.at(g[0::3], i_idx, penalty * gx)
        np.add.at(g[0::3], j_idx, -penalty * gx)
        
        np.add.at(g[1::3], i_idx, penalty * gy)
        np.add.at(g[1::3], j_idx, -penalty * gy)
        
        np.add.at(g[2::3], i_idx, penalty * gr)
        np.add.at(g[2::3], j_idx, penalty * gr)

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


def _slsqp_polish(p, time_limit=5.0):
    """
    Final polish using SLSQP with hard constraints.
    """
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    
    # Build constraints
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
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - p[3*i+2] - p[3*j+2]
        cons.append({'type': 'ineq', 'fun': pair_fun})
    
    def obj(params):
        return -np.sum(params[2::3])
    
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
            options={'maxiter': 50, 'ftol': 1e-12}
        )
        
        if res.success:
            new_p = _enforce_validity(res.x.reshape(N, 3))
            new_score = _score_valid(new_p)
            if new_score > best_score:
                best_score = new_score
                best_p = new_p
    except Exception:
        pass
    
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
        _hex_initial,
        _random_initial,
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
            p0 = _enforce_validity(x0)
            
            # Phase 1: L-BFGS-B with Delaunay smooth penalty
            # We need to re-compute edges periodically or just use all pairs if N is small enough?
            # N=26, pairs=325. Delaunay is ~ 3*N = 78 edges. Much faster.
            # However, Delaunay changes. We can do a few iterations, re-compute, then continue.
            # For simplicity and robustness in the time limit, we'll run L-BFGS-B with the *current* edges.
            # If the geometry changes drastically, the edges might be wrong, but the smooth penalty
            # will still push non-overlapping circles apart if they are close, and the wall constraints
            # are global. The main risk is missing a constraint between two circles that become
            # neighbors later. But since we start from a valid config and move slowly, this is less likely.
            # To be safer, we can re-compute edges every 10 iterations if we could hook in, but scipy minimize
            # doesn't easily allow that. 
            # Alternative: Use all pairs for the penalty but it's slower. 
            # Let's stick to Delaunay for speed, but if the score doesn't improve, we might need to be careful.
            
            # Actually, a safer bet for correctness within the time limit is to use the full pairwise penalty
            # if N is small (26 is small). 325 pairs is not that many.
            # Let's try full pairwise smooth penalty first for correctness, as Delaunay might miss constraints
            # if the initial graph is bad.
            
            # Let's create a "full" edge list for the first run to ensure safety, then maybe switch?
            # No, let's just use full pairwise for the smooth objective. It's 325 ops, very fast.
            full_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
            
            try:
                res = minimize(
                    _delaunay_objective,
                    p0.flatten(),
                    method='L-BFGS-B',
                    jac=_delaunay_grad,
                    bounds=bounds_list,
                    args=(full_edges, 1000.0, 1e-4),
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                p = _enforce_validity(res.x.reshape(N, 3))
                s = _score_valid(p)
                
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
                    
            except Exception:
                continue
            
            # Phase 2: SLSQP Polish
            if time.time() > deadline - 20:
                break
                
            if best_params is not None:
                try:
                    p_refined, s_refined = _slsqp_polish(
                        best_params, 
                        time_limit=min(5.0, max(0.1, deadline - time.time() - 5))
                    )
                    if s_refined > best_score:
                        best_score = s_refined
                        best_params = p_refined.copy()
                except Exception:
                    continue

    # If we didn't find any valid solution, use a fallback
    if best_params is None:
        best_params = _enforce_validity(_hex_initial(42)).reshape(N, 3)

    # Final refinement: Shake and Refine
    if time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        current_best = best_params.copy()
        current_score = best_score
        sigma = 0.02
        
        while time.time() < deadline - 5:
            p_new = current_best.copy()
            p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
            p_new[:, 2] += rng.normal(0, sigma * 0.5, size=N)
            
            p_new = _enforce_validity(p_new)
            
            # Quick L-BFGS-B on the shaken config
            full_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
            try:
                res = minimize(
                    _delaunay_objective,
                    p_new.flatten(),
                    method='L-BFGS-B',
                    jac=_delaunay_grad,
                    bounds=bounds_list,
                    args=(full_edges, 1000.0, 1e-4),
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                p_opt = _enforce_validity(res.x.reshape(N, 3))
                s_opt = _score_valid(p_opt)
                
                if s_opt > current_score:
                    current_score = s_opt
                    current_best = p_opt
                else:
                    # Decay sigma if no improvement
                    sigma *= 0.9
            except Exception:
                pass

        if current_score > best_score:
            best_score = current_score
            best_params = current_best

    # Return in the required format: (x, y, radius)
    result = best_params.copy()
    return result

