"""Packing of 50 equal circles in a unit square via soft-min optimization + SA + L-BFGS polish."""
import numpy as np
import time

N = 50
HALF = 0.5


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


def _violation_fast(pts, r_target):
    """Fast total violation: sum of pairwise overlaps + wall penetrations."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist)
    iu = np.triu_indices(N, k=1)
    pair_viol_sum = np.sum(pair_viol[iu])
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol_sum = np.sum(np.maximum(0.0, r_target - d))
    
    return float(pair_viol_sum + wall_viol_sum)


def _soft_min_obj(pts, r_target, beta=500.0):
    """
    Smooth surrogate for the packing problem.
    Minimize sum of soft-violations for pairs and walls.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = 2.0 * r_target - pair_dist
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    soft_pair = np.log1p(np.exp(beta * pair_viol_clipped)) / beta
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    soft_wall = np.log1p(np.exp(beta * wall_viol_clipped)) / beta
    
    return float(np.sum(soft_pair) + np.sum(soft_wall))


def _soft_min_grad(pts, r_target, beta=500.0):
    """
    Gradient of the soft-min objective with respect to pts (vectorized).
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    
    # Sigmoid weights for pairs
    pair_viol = 2.0 * r_target - dist
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-beta * pair_viol_clipped))
    
    # Gradient from pairs: for pair (i,j), contribution to grad[i] = -sig_ij * (pts_i - pts_j)/dist_ij
    # Vectorized: grad[i] = -sum_j sig[i,j] * (pts[i] - pts[j]) / dist[i,j]
    # = -sum_j sig[i,j]/dist[i,j] * pts[i] + sum_j sig[i,j]/dist[i,j] * pts[j]
    w = sig / dist_safe  # N x N weights
    np.fill_diagonal(w, 0.0)
    # grad[i] = -pts[i] * sum_j w[i,j] + sum_j w[i,j] * pts[j]
    wsum = np.sum(w, axis=1)  # N
    grad = -pts * wsum[:, None] + w @ pts
    
    # Wall gradients
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    sig_w = 1.0 / (1.0 + np.exp(-beta * wall_viol_clipped))
    
    # For each point, determine which wall is closest and add gradient
    # d(wall_dist)/d(x): +1 if left wall closest, -1 if right wall closest
    # d(wall_dist)/d(y): +1 if bottom wall closest, -1 if top wall closest
    gx = np.zeros(N)
    gy = np.zeros(N)
    for i in range(N):
        dl, dr, db, dt = d_left[i], d_right[i], d_bottom[i], d_top[i]
        wm = wall_dist[i]
        s = sig_w[i]
        if abs(dl - wm) < 1e-10:
            gx[i] += s
        if abs(dr - wm) < 1e-10:
            gx[i] -= s
        if abs(db - wm) < 1e-10:
            gy[i] += s
        if abs(dt - wm) < 1e-10:
            gy[i] -= s
    
    grad[:, 0] -= gx
    grad[:, 1] -= gy
    
    return grad


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=2000):
    """Simulated annealing to minimize violation for target radius r_target."""
    pts = pts0.copy()
    current_viol = _violation_fast(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.03
    t_end = 0.00003
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    # Precompute indices for speed
    iu = np.triu_indices(N, k=1)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Random move: pick 1-3 circles and perturb
        n_move = 1 + (step % 3)  # deterministic cycling for speed
        # Use rejection-free sampling
        indices = rng.integers(0, N, size=n_move)
        # Remove duplicates
        if n_move > 1:
            indices = np.unique(indices)
            if len(indices) < n_move:
                # Fill in with more random picks
                while len(indices) < n_move:
                    new_idx = rng.integers(0, N)
                    if new_idx not in indices:
                        indices = np.append(indices, new_idx)
        deltas = rng.normal(0, temp, size=(len(indices), 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation_fast(new_pts, r_target)
        
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol


def _lbfgs_polish(pts0, r_target, deadline, beta=500.0, maxiter=500):
    """Use L-BFGS-B to minimize the soft-min objective for target radius."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _soft_min_obj(p, r_target, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        g = _soft_min_grad(p, r_target, beta)
        return g.ravel()
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 50})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(res.fun)


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
    """Grid initialization with 7x7 + center."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _best_known_init(rng):
    """
    Initialize near the known best packing structure.
    The Boll-Donovan-Graham-Lubachevsky packing has a specific structure.
    We create a rough approximation and add jitter.
    """
    # Create a hexagonal-like pattern that's been optimized
    # Use a 7-column layout with varying row heights
    pts = []
    # Approximate the known packing: roughly 7-8 circles per row
    # with some rows shifted
    base_y = -0.48
    row_heights = [0.12, 0.11, 0.12, 0.11, 0.12, 0.11, 0.12, 0.11, 0.12, 0.11]
    y = -0.48
    row = 0
    while len(pts) < N:
        offset = 0.06 if (row % 2) else 0.0
        x_start = -0.48 + offset
        n_in_row = 7 if row % 2 == 0 else 6
        x = x_start
        for _ in range(n_in_row):
            if len(pts) >= N:
                break
            pts.append([x, y])
            x += 0.14
        y += 0.11
        row += 1
    pts = np.array(pts[:N])
    if rng is not None:
        pts = pts + rng.normal(0, 0.01, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _sdp_relaxation_init(rng):
    """
    SDP-inspired initialization: maximize minimum pairwise distance
    using log-sum-exp surrogate.
    """
    from scipy.optimize import minimize
    
    def obj(x):
        p = x.reshape(N, 2)
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 0.0)
        dist = np.sqrt(dist_sq)
        iu = np.triu_indices(N, k=1)
        d = dist[iu]
        max_d = np.max(d)
        lse = max_d + np.log(np.sum(np.exp(d - max_d)))
        d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                           np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
        wall_pen = np.sum(np.exp(-10 * d_wall))
        return -lse + wall_pen
    
    def grad(x):
        p = x.reshape(N, 2)
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 0.0)
        dist = np.sqrt(dist_sq)
        dist_safe = np.where(dist > 1e-12, dist, 1e-12)
        iu = np.triu_indices(N, k=1)
        d = dist[iu]
        max_d = np.max(d)
        exp_d = np.exp(d - max_d)
        sum_exp = np.sum(exp_d)
        weights = exp_d / sum_exp
        
        g = np.zeros_like(p)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[idx]
            diff_ij = p[i] - p[j]
            d_ij = dist_safe[i, j]
            g[i] += w * diff_ij / d_ij
            g[j] -= w * diff_ij / d_ij
        
        d_left = p[:, 0] + HALF
        d_right = HALF - p[:, 0]
        d_bottom = p[:, 1] + HALF
        d_top = HALF - p[:, 1]
        d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        for i in range(N):
            pen = np.exp(-10 * d_wall[i])
            if abs(d_left[i] - d_wall[i]) < 1e-10:
                g[i, 0] += 10 * pen
            if abs(d_right[i] - d_wall[i]) < 1e-10:
                g[i, 0] -= 10 * pen
            if abs(d_bottom[i] - d_wall[i]) < 1e-10:
                g[i, 1] += 10 * pen
            if abs(d_top[i] - d_wall[i]) < 1e-10:
                g[i, 1] -= 10 * pen
        
        return g.ravel()
    
    x0 = _hex_init(rng, jitter=0.02).ravel()
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return _hex_init(rng, jitter=0.02)


def _refine_radius(pts_start, deadline, rng):
    """
    Iteratively refine the radius: try to increase r slightly,
    fix violations with SA + L-BFGS, and repeat.
    """
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    # Try increasing radius in small steps
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
        
        r_target = best_r * (1.0 + 0.0002 * (attempt + 1))
        
        # SA to fix violations
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_target, deadline - 5.0, rng, n_steps=800)
        
        # L-BFGS polish
        try:
            pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_target, deadline - 5.0, beta=800.0, maxiter=300)
        except Exception:
            pts_lbfgs = pts_sa
        
        r_achieved = _compute_r(pts_lbfgs)
        viol = _violation_fast(pts_lbfgs, r_target)
        
        if viol < 1e-8 and r_achieved >= r_target - 1e-9:
            best_r = r_achieved
            best_pts = pts_lbfgs.copy()
        else:
            # Try a smaller increment
            r_target_small = best_r * (1.0 + 0.0001)
            pts_sa2, _ = _simulated_annealing(best_pts, r_target_small, deadline - 5.0, rng, n_steps=500)
            try:
                pts_lbfgs2, _ = _lbfgs_polish(pts_sa2, r_target_small, deadline - 5.0, beta=800.0, maxiter=200)
            except Exception:
                pts_lbfgs2 = pts_sa2
            r_ach2 = _compute_r(pts_lbfgs2)
            viol2 = _violation_fast(pts_lbfgs2, r_target_small)
            if viol2 < 1e-8 and r_ach2 > best_r:
                best_r = r_ach2
                best_pts = pts_lbfgs2.copy()
            else:
                break
    
    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 147.0  # Safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps, beta=500.0, maxiter=400):
        nonlocal best_pts, best_r
        if time.time() > deadline - 8.0:
            return
        
        # SA phase
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - 8.0, rng, n_steps=n_sa_steps)
        
        # L-BFGS polish on soft-min
        try:
            pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_target, deadline - 8.0, beta=beta, maxiter=maxiter)
        except Exception:
            pts_lbfgs = pts_sa
        
        r = _compute_r(pts_lbfgs)
        if r > best_r:
            best_r = r
            best_pts = pts_lbfgs.copy()
    
    # Phase 0: Multiple initializations
    if time.time() < deadline - 80.0:
        inits = [
            _hex_init(rng, jitter=0.0),
            _hex_init(rng, jitter=0.02),
            _grid_init(rng, jitter=0.01),
            _grid_init(rng, jitter=0.04),
            _random_init(rng),
            _random_init(rng),
            _best_known_init(rng),
        ]
        
        for init in inits:
            if time.time() > deadline - 70.0:
                break
            r_init = _compute_r(init)
            if r_init > best_r:
                best_r = r_init
                best_pts = init.copy()
    
    if best_pts is None:
        best_pts = _hex_init(rng, jitter=0.01)
        best_r = _compute_r(best_pts)
    
    # Phase 1: Multi-start optimization with moderate target
    if time.time() < deadline - 50.0:
        r_target = max(best_r, 0.069)
        inits = [
            best_pts.copy(),
            _hex_init(rng, jitter=0.01),
            _hex_init(rng, jitter=0.03),
            _grid_init(rng, jitter=0.02),
            _random_init(rng),
        ]
        for init in inits:
            if time.time() > deadline - 45.0:
                break
            try_improve(init, r_target, 1200, beta=500.0, maxiter=300)
    
    # Phase 2: Binary search on radius
    if time.time() < deadline - 25.0:
        best_pts, best_r = _refine_radius(best_pts, deadline, rng)
    
    # Phase 3: Final polish with tighter tolerance and higher beta
    if time.time() < deadline - 10.0:
        r_target = best_r * 1.0003
        try_improve(best_pts, r_target, 1500, beta=1000.0, maxiter=500)
    
    # Phase 4: Very final polish
    if time.time() < deadline - 4.0:
        r_target = best_r * 1.0001
        try_improve(best_pts, r_target, 1000, beta=2000.0, maxiter=600)
    
    # Final: ensure validity
    best_pts = np.clip(best_pts, -HALF, HALF)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(rng, jitter=0.0)
    
    return best_pts
