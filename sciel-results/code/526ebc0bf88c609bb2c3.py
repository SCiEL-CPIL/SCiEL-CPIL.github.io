"""Packing of 50 equal circles in a unit square via vectorized optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
N_PAIRS = len(IU[0])

def _compute_radius(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r_wall = float(np.min(d_wall))
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = float(np.min(dist[IU])) / 2.0
    
    return min(r_wall, r_pair)

def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations. Vectorized."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Pairwise violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist[IU])
    
    # Wall violations
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d_wall)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))

def _violation_grad(pts, r_target):
    """Analytical gradient of violation for target radius. Vectorized."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    grad = np.zeros((N, 2))
    
    # Pairwise gradient
    mask = dist < 2.0 * r_target
    safe_dist = np.where(mask, dist, 1.0)
    
    unit_vecs = diff / safe_dist[:, :, None]
    unit_vecs = unit_vecs * mask[:, :, None]
    grad_pair = -np.sum(unit_vecs, axis=1)
    
    grad += grad_pair
    
    # Wall gradient
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    mask_left = d_left < r_target
    mask_right = d_right < r_target
    mask_bottom = d_bottom < r_target
    mask_top = d_top < r_target
    
    grad[:, 0] += mask_left.astype(float)
    grad[:, 0] -= mask_right.astype(float)
    grad[:, 1] += mask_bottom.astype(float)
    grad[:, 1] -= mask_top.astype(float)
    
    return grad

def _soft_min_radius(pts, beta=50.0):
    """Compute soft-min radius estimate."""
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = max_d - log_sum / beta
    
    return r_soft

def _soft_min_obj(x, beta=50.0):
    """Objective: negative soft-min radius (to minimize)."""
    p = x.reshape(N, 2)
    return -_soft_min_radius(p, beta)

def _soft_min_grad(x, beta=50.0):
    """Gradient of negative soft-min radius."""
    p = x.reshape(N, 2)
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Wall gradients
    d_left = p[:, 0] + HALF
    d_right = HALF - p[:, 0]
    d_bottom = p[:, 1] + HALF
    d_top = HALF - p[:, 1]
    
    walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    min_idx = np.argmin(walls, axis=1)
    
    for i in range(N):
        w = weights[i]
        if w > 1e-10:
            if min_idx[i] == 0: grad[i, 0] += w
            elif min_idx[i] == 1: grad[i, 0] -= w
            elif min_idx[i] == 2: grad[i, 1] += w
            else: grad[i, 1] -= w
            
    # Pair gradients
    for k in range(N_PAIRS):
        i, j = IU[0][k], IU[1][k]
        w = weights[N + k]
        if w > 1e-10:
            dij = p[i] - p[j]
            dist_ij = np.linalg.norm(dij)
            if dist_ij > 1e-10:
                g = w * dij / (2.0 * dist_ij)
                grad[i] += g
                grad[j] -= g
                
    return -grad.ravel()

def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=8000):
    """Fast simulated annealing to minimize violation."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.015
    t_end = 0.00005
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        temp = t_start * (t_end / t_start) ** progress
        
        # Move 1 or 2 random circles
        n_move = rng.integers(1, 3)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol

def _lbfgs_polish(pts0, r_target, deadline):
    """Polish using L-BFGS-B to minimize violation for target radius."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation(p, r_target)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation_grad(p, r_target).ravel()
    
    try:
        res = minimize(obj, pts0.ravel(), method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts, float(res.fun)
    except Exception:
        return pts0, _violation(pts0, r_target)

def _soft_min_polish(pts0, deadline, beta_start=30.0, beta_end=500.0, n_iters=5):
    """Polish using soft-min radius maximization with increasing beta."""
    best_pts = pts0.copy()
    best_r = _compute_radius(pts0)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        # Geometric progression for beta
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iters - 1, 1))
        
        def obj(x):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            return _soft_min_obj(x, beta)
        
        def grad(x):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            return _soft_min_grad(x, beta)
        
        try:
            res = minimize(obj, best_pts.ravel(), method='L-BFGS-B', jac=grad,
                           bounds=[(-HALF, HALF)] * (2 * N),
                           options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14})
            pts = res.x.reshape(N, 2)
            pts = np.clip(pts, -HALF, HALF)
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        except Exception:
            pass
    
    return best_pts, best_r

def _contact_local_search(pts0, r_target, deadline, rng, n_steps=3000):
    """Local search focusing on contact graph (touching circles)."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Find circles that are in violation (touching or overlapping)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                            np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
        
        # Circles that are too close to each other or walls
        touching_pairs = dist < 2.0 * r_target * 1.01
        touching_walls = d_wall < r_target * 1.01
        
        # Get indices of circles involved in violations
        involved = np.zeros(N, dtype=bool)
        involved[touching_walls] = True
        if np.any(touching_pairs):
            involved[np.where(touching_pairs)[0]] = True
            involved[np.where(touching_pairs)[1]] = True
        
        if not np.any(involved):
            # No violations, try random small moves
            indices = rng.choice(N, size=1, replace=False)
        else:
            # Focus on involved circles
            involved_idx = np.where(involved)[0]
            n_move = min(rng.integers(1, 3), len(involved_idx))
            indices = rng.choice(involved_idx, size=n_move, replace=False)
        
        # Small moves for local refinement
        step_size = 0.005 * (1.0 - step / n_steps) + 0.001
        deltas = rng.normal(0, step_size, size=(len(indices), 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        delta = new_viol - current_viol
        if delta < 0:
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol

def _hex_init(rng, jitter=0.0):
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def construct():
    start = time.time()
    deadline = start + 140.0
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, deadline_offset=5.0):
        nonlocal best_pts, best_r
        if time.time() > deadline - deadline_offset:
            return
        
        # Phase 1: SA to get close to feasible
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - deadline_offset, rng, n_steps=5000)
        
        # Phase 2: Contact local search for refinement
        pts_cl, viol_cl = _contact_local_search(pts_sa, r_target, deadline - deadline_offset, rng, n_steps=2000)
        
        # Phase 3: L-BFGS polish
        pts_lbfgs, viol_lbfgs = _lbfgs_polish(pts_cl, r_target, deadline - deadline_offset)
        
        # Phase 4: Soft-min polish to push radius
        pts_soft, r_soft = _soft_min_polish(pts_lbfgs, deadline - deadline_offset)
        
        r = _compute_radius(pts_soft)
        if r > best_r:
            best_r = r
            best_pts = pts_soft.copy()
    
    # Initializations - diverse starts
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
    ]
    
    # Phase 1: Get a good starting point with moderate radius
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        try_improve(init, 0.07, deadline_offset=8.0)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search to push radius higher
    r_lo = best_r
    r_hi = 0.07138
    
    n_push_iters = 12
    for i in range(n_push_iters):
        if time.time() > deadline - 25.0:
            break
        r_mid = (r_lo + r_hi) / 2.0
        try_improve(best_pts, r_mid, deadline_offset=10.0)
        
        r_check = _compute_radius(best_pts)
        if r_check >= r_mid - 1e-9:
            r_lo = r_mid
        else:
            r_hi = r_mid
    
    # Phase 3: Final high-precision soft-min polish with very high beta
    if time.time() < deadline - 8.0:
        # Multiple passes with increasing beta for ultra-tight convergence
        for beta in [100.0, 300.0, 1000.0, 5000.0]:
            if time.time() > deadline - 5.0:
                break
            
            def obj(x):
                if time.time() > deadline - 3.0:
                    raise RuntimeError("timeout")
                return _soft_min_obj(x, beta)
            
            def grad(x):
                if time.time() > deadline - 3.0:
                    raise RuntimeError("timeout")
                return _soft_min_grad(x, beta)
            
            try:
                res = minimize(obj, best_pts.ravel(), method='L-BFGS-B', jac=grad,
                               bounds=[(-HALF, HALF)] * (2 * N),
                               options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
                pts_final = res.x.reshape(N, 2)
                pts_final = np.clip(pts_final, -HALF, HALF)
                r_final = _compute_radius(pts_final)
                if r_final > best_r:
                    best_r = r_final
                    best_pts = pts_final
            except Exception:
                pass
            
    # Final safety check
    if best_pts is not None:
        final_r = _compute_radius(best_pts)
        if final_r > 0:
            return best_pts
    
    # Fallback
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
