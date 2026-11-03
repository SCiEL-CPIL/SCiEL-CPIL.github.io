"""
Packing 50 equal circles in a unit square.
Strategy:
  1. Generate diverse initializations.
  2. Fast local search on each init to get good starting points for SLSQP.
  3. SLSQP polish on the best candidate with a strict time limit.
  4. Return the best result found within the time budget.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
KNOWN_R = 0.071377103865

def _true_radius(centers):
    """Compute the true maximum packing radius for given centers in [-0.5,0.5]^2."""
    centers = np.asarray(centers)
    n = centers.shape[0]
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _local_search(centers, max_iter=1000, time_limit=5.0, step_init=0.015):
    """
    Fast hill-climbing local search.
    Returns (centers, best_radius).
    """
    centers = centers.copy()
    best_r = _true_radius(centers)
    step_size = step_init
    no_improve = 0
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        indices = np.random.permutation(N)
        for idx in indices:
            orig = centers[idx].copy()
            # Current local min for this circle
            wall_d = 0.5 - np.abs(orig)
            min_wall = np.min(wall_d)
            diffs = centers - orig
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            dists[idx] = np.inf
            min_pair = np.min(dists) / 2.0
            cur_local = min(min_wall, min_pair)
            
            # Determine push direction: away from closest constraint
            closest_idx = np.argmin(dists)
            if dists[closest_idx] / 2.0 < min_wall:
                direction = orig - centers[closest_idx]
            else:
                # Move toward center (away from nearest wall)
                direction = -orig
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = np.random.normal(0, 1, 2)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Try several perturbations
            best_local = cur_local
            best_new = orig.copy()
            for _ in range(10):
                noise = np.random.normal(0, 1, 2)
                move = direction * step_size + noise * step_size * 0.4
                new_c = np.clip(orig + move, -0.4999, 0.4999)
                # Quick local min
                nw = 0.5 - np.abs(new_c)
                nmin_wall = np.min(nw)
                nd = centers - new_c
                ndists = np.sqrt(np.sum(nd ** 2, axis=1))
                ndists[idx] = np.inf
                nmin_pair = np.min(ndists) / 2.0
                nlocal = min(nmin_wall, nmin_pair)
                if nlocal > best_local + 1e-12:
                    best_local = nlocal
                    best_new = new_c
            if best_local > cur_local + 1e-12:
                centers[idx] = best_new
                improved = True
        if not improved:
            no_improve += 1
            step_size *= 0.9
            if no_improve > 20:
                break
        else:
            no_improve = 0
        r = _true_radius(centers)
        if r > best_r:
            best_r = r
    return centers, best_r

def _slsqp_polish(centers, time_limit=30.0):
    """
    Polish with SLSQP directly maximizing r subject to hard constraints.
    """
    start = time.time()
    best_c = centers.copy()
    best_r = _true_radius(centers)
    
    def objective(v):
        return -v[-1]  # maximize r
    
    def gradient(v):
        g = np.zeros_like(v)
        g[-1] = -1.0
        return g
    
    # Constraints: ||c_i - c_j||^2 - 4*r^2 >= 0
    def make_constraints():
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                ii, jj = 2*i, 2*j
                def constraint(v, ii=ii, jj=jj):
                    dx = v[ii] - v[jj]
                    dy = v[ii+1] - v[jj+1]
                    r = v[-1]
                    return dx*dx + dy*dy - 4*r*r
                cons.append({'type': 'ineq', 'fun': constraint})
        return cons
    
    constraints = make_constraints()
    
    # Bounds: coordinates in [-0.5, 0.5], r >= 0
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)]
    
    # Initial guess for r
    r0 = _true_radius(centers)
    v0 = np.concatenate([centers.flatten(), [r0]])
    
    try:
        result = minimize(
            objective,
            v0,
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        new_c = result.x[:2*N].reshape(N, 2)
        new_r = result.x[-1]
        # Clip to valid range
        new_c = np.clip(new_c, -0.499999, 0.499999)
        true_r = _true_radius(new_c)
        if true_r > best_r:
            best_r = true_r
            best_c = new_c
    except Exception:
        pass
    
    return best_c, best_r

def _hex_init(seed=42):
    """Hexagonal grid init."""
    np.random.seed(seed)
    s = 0.14
    pts = []
    for i, x in enumerate(np.arange(-0.8, 0.8, s)):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in np.arange(-0.8, 0.8, s):
            pts.append((x + offset, y))
    pts = np.array(pts)
    mask = (np.abs(pts[:, 0]) <= 0.49) & (np.abs(pts[:, 1]) <= 0.49)
    pts = pts[mask]
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()

def _square_grid_init(seed=123):
    """Square grid init."""
    np.random.seed(seed)
    gx = np.linspace(-0.45, 0.45, 7)
    gy = np.linspace(-0.45, 0.45, 7)
    pts = np.array([[x, y] for x in gx for y in gy])
    if len(pts) < N:
        return np.random.uniform(-0.45, 0.45, (N, 2))
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()

def _random_init(seed=999):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _jittered_hex(seed=43):
    base = _hex_init(seed)
    np.random.seed(seed)
    jittered = base + np.random.normal(0, 0.02, (N, 2))
    return np.clip(jittered, -0.49, 0.49)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    inits = [
        _hex_init(42),
        _square_grid_init(123),
        _random_init(999),
        _jittered_hex(43),
    ]
    
    best_pts = inits[0].copy()
    best_r = _true_radius(best_pts)
    
    # Phase 1: Fast local search on each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        ls_c, ls_r = _local_search(init, max_iter=1000, time_limit=5.0)
        if ls_r > best_r:
            best_r = ls_r
            best_pts = ls_c.copy()
    
    # Phase 2: SLSQP polish on best
    if time.time() < deadline - 5.0:
        slsqp_c, slsqp_r = _slsqp_polish(best_pts, time_limit=30.0)
        if slsqp_r > best_r:
            best_r = slsqp_r
            best_pts = slsqp_c.copy()
    
    best_pts = np.clip(best_pts, -0.499999999, 0.499999999)
    return best_pts
