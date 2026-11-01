"""Packing of 50 equal circles in a unit square via soft-min optimization + SA + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
N_PAIRS = len(IU[0])


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _soft_min_obj(x, beta):
    """
    Objective: Minimize -softmin(distances).
    This is equivalent to maximizing the softmin of distances.
    softmin(d) = -1/beta * log(sum(exp(-beta * d)))
    We want to maximize min(d), so we minimize -softmin(d).
    """
    p = x.reshape(N, 2)
    
    # Wall distances
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    
    # Pair distances (halved, as radius is half the distance)
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    
    # Numerical stability: shift by max
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = max_d - log_sum / beta
    
    return -r_soft


def _soft_min_grad(x, beta):
    """
    Gradient of -softmin(distances).
    d/dx [-softmin(d)] = -d/dx [max_d - log(sum(exp(-beta*(d-max_d))))/beta]
    Note: max_d is constant w.r.t x in the gradient calculation if we treat it as a shift, 
    but strictly speaking, the softmin gradient is:
    g_i = - sum_j w_j * grad(d_j)
    where w_j = exp(-beta * d_j) / sum(exp(-beta * d_k))
    """
    p = x.reshape(N, 2)
    
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    
    # Weights
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Wall gradients
    # d(wall_dist)/d(p_i)
    d_left = p[:, 0] + HALF
    d_right = HALF - p[:, 0]
    d_bottom = p[:, 1] + HALF
    d_top = HALF - p[:, 1]
    walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    min_idx = np.argmin(walls, axis=1)
    
    for i in range(N):
        w = weights[i]
        if w > 1e-10:
            if min_idx[i] == 0: grad[i, 0] += w      # d(x+0.5)/dx = 1
            elif min_idx[i] == 1: grad[i, 0] -= w    # d(0.5-x)/dx = -1
            elif min_idx[i] == 2: grad[i, 1] += w    # d(y+0.5)/dy = 1
            else: grad[i, 1] -= w                    # d(0.5-y)/dy = -1
    
    # Pair gradients
    # d(pair_d_k)/d(p_i) for pair (i,j)
    # pair_d = dist/2
    # d(dist)/d(p_i) = (p_i - p_j) / dist
    # d(pair_d)/d(p_i) = (p_i - p_j) / (2 * dist)
    
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
    
    # The objective is -softmin, so we negate the gradient of softmin
    # Wait, let's re-verify the sign.
    # Obj = -softmin.
    # softmin = max_d - (1/beta) log sum exp(-beta (d - max_d))
    # d(softmin)/dx = - (1/beta) * (1/sum) * sum(-beta exp(-beta(d-max_d)) * d(d)/dx)
    # = sum(exp(-beta(d-max_d))/sum) * d(d)/dx = sum(w_k * d(d_k)/dx)
    # So d(Obj)/dx = - sum(w_k * d(d_k)/dx)
    
    return -grad.ravel()


def _violation(pts, r_target):
    """Total violation for SA."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist[IU])
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=3000):
    """SA to minimize hard violation for target radius."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.04
    t_end = 0.00005
    cooling = (t_end / t_start) ** (1.0 / max(n_steps - 1, 1))
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        n_move = rng.integers(1, 4)
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


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
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
    deadline = start + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Multi-start with diverse initializations
    # We use a soft-min objective to maximize radius directly.
    # This avoids the need to guess a target radius.
    
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.0),
        _grid_init(rng, jitter=0.02),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Polish with L-BFGS on the soft-min objective
        # Start with a low beta (smooth) and increase for precision
        current_pts = init
        for beta in [10.0, 50.0, 200.0, 1000.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(_soft_min_obj, current_pts.ravel(), method='L-BFGS-B',
                               jac=_soft_min_grad, args=(beta,),
                               bounds=[(-HALF, HALF)] * (2 * N),
                               options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12})
                pts_polished = res.x.reshape(N, 2)
                pts_polished = np.clip(pts_polished, -HALF, HALF)
                
                r = _compute_r(pts_polished)
                if r > best_r:
                    best_r = r
                    best_pts = pts_polished.copy()
                current_pts = pts_polished
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Refine with SA + Polish loop
    # Use the best point found so far as a seed for SA to try to escape local minima
    if time.time() < deadline - 10.0:
        r_target = best_r + 0.0001
        pts_sa, _ = _simulated_annealing(best_pts, r_target, deadline - 5.0, rng, n_steps=1500)
        
        # Polish the SA result
        for beta in [100.0, 500.0, 2000.0]:
            if time.time() > deadline - 2.0:
                break
            try:
                res = minimize(_soft_min_obj, pts_sa.ravel(), method='L-BFGS-B',
                               jac=_soft_min_grad, args=(beta,),
                               bounds=[(-HALF, HALF)] * (2 * N),
                               options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
                pts_final = res.x.reshape(N, 2)
                pts_final = np.clip(pts_final, -HALF, HALF)
                r_final = _compute_r(pts_final)
                if r_final > best_r:
                    best_r = r_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    return best_pts
