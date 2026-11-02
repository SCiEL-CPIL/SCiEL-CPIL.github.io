"""Packing of 50 equal circles in a unit square via soft-min optimization + contact-graph search."""
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
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r_wall = float(np.min(d_wall))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = float(np.min(dist[IU])) / 2.0
    return min(r_wall, r_pair)

def _soft_min_obj(x, beta):
    """Negative soft-min of all constraint distances (wall + pair/2)."""
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
    log_sum = np.log(np.sum(exp_terms))
    r_soft = max_d - log_sum / beta
    return -r_soft

def _soft_min_grad(x, beta):
    """Gradient of negative soft-min."""
    p = x.reshape(N, 2)
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    all_d = np.concatenate([d_wall, pair_d])
    max_d = np.max(all_d)
    # Use numerically stable computation for high beta
    shifted = all_d - max_d
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    grad = np.zeros((N, 2))
    # Wall gradients: for each point, find which wall is closest
    d_left = p[:, 0] + HALF
    d_right = HALF - p[:, 0]
    d_bottom = p[:, 1] + HALF
    d_top = HALF - p[:, 1]
    walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    min_idx = np.argmin(walls, axis=1)
    w_wall = weights[:N]
    # Vectorized wall gradient computation
    for i in range(N):
        w = w_wall[i]
        if w > 1e-15:
            if min_idx[i] == 0: grad[i, 0] += w
            elif min_idx[i] == 1: grad[i, 0] -= w
            elif min_idx[i] == 2: grad[i, 1] += w
            else: grad[i, 1] -= w
    # Pair gradients - vectorized
    w_pair = weights[N:]
    i_idx = IU[0]
    j_idx = IU[1]
    dij = p[i_idx] - p[j_idx]
    dist_ij = np.linalg.norm(dij, axis=1)
    safe_dist = np.where(dist_ij > 1e-15, dist_ij, 1.0)
    unit_vecs = dij / safe_dist[:, None]
    g = (w_pair / (2.0 * safe_dist))[:, None] * unit_vecs
    # Accumulate gradients
    for k in range(N_PAIRS):
        if w_pair[k] > 1e-15:
            grad[i_idx[k]] += g[k]
            grad[j_idx[k]] -= g[k]
    return -grad.ravel()

def _violation(pts, r_target):
    """Sum of violations (pair overlaps + wall penetrations)."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist[IU])
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d_wall)
    return float(np.sum(pair_viol) + np.sum(wall_viol))

def _contact_graph(pts, r_target, margin=0.02):
    """Build contact graph: pairs within (2r + margin) and points near walls."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    threshold = 2.0 * r_target + margin
    pairs = np.argwhere(dist < threshold)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    # Wall contacts
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_pts = np.where(d_wall < r_target + margin)[0]
    return pairs, wall_pts

def _contact_graph_local_search(pts, r_target, deadline, rng, n_iters=2000):
    """Local search on contact graph: push apart overlapping pairs, push from walls."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_viol = _violation(pts, r_target)
    
    step_size = 0.005
    for it in range(n_iters):
        if time.time() > deadline:
            break
        pairs, wall_pts = _contact_graph(pts, r_target, margin=0.03)
        
        # Compute forces
        forces = np.zeros((N, 2))
        for i, j in pairs:
            dij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(dij)
            if dist_ij > 1e-10:
                overlap = 2.0 * r_target - dist_ij
                if overlap > 0:
                    f = overlap * dij / dist_ij
                    forces[i] += f
                    forces[j] -= f
        for i in wall_pts:
            # Push away from nearest wall
            dx = pts[i, 0]
            dy = pts[i, 1]
            dl = dx + HALF
            dr = HALF - dx
            db = dy + HALF
            dt = HALF - dy
            min_d = min(dl, dr, db, dt)
            if min_d < r_target:
                pen = r_target - min_d
                if min_d == dl: forces[i, 0] += pen
                elif min_d == dr: forces[i, 0] -= pen
                elif min_d == db: forces[i, 1] += pen
                else: forces[i, 1] -= pen
        
        # Apply forces with step size, add small random perturbation
        step = step_size * forces + rng.normal(0, 0.001, size=(N, 2))
        new_pts = pts + step
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        if new_viol < best_viol:
            best_viol = new_viol
            best_pts = new_pts.copy()
        # Accept if improvement or small probability
        if new_viol <= _violation(pts, r_target) * 1.001 or rng.random() < 0.1:
            pts = new_pts
        else:
            pts = best_pts.copy()
        
        # Gradually reduce step size
        if it % 200 == 199:
            step_size *= 0.8
    
    return best_pts, best_viol

def _lbfgs_softmin(pts0, beta, deadline):
    """Polish using L-BFGS-B on soft-min objective."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_min_obj(x, beta)
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_min_grad(x, beta)
    try:
        # Use tighter tolerances and more iterations for high-beta polishing
        maxiter = 400 if beta > 500 else 300
        ftol = 1e-18 if beta > 500 else 1e-16
        gtol = 1e-14 if beta > 500 else 1e-12
        res = minimize(obj, pts0.ravel(), method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 60})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return pts0

def _lbfgs_violation(pts0, r_target, deadline):
    """Polish using L-BFGS-B on violation objective."""
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
                       options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 20})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return pts0

def _violation_grad(pts, r_target):
    """Gradient of violation function."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    grad = np.zeros((N, 2))
    mask = dist < 2.0 * r_target
    safe_dist = np.where(mask, dist, 1.0)
    unit_vecs = diff / safe_dist[:, :, None]
    unit_vecs = unit_vecs * mask[:, :, None]
    grad_pair = -np.sum(unit_vecs, axis=1)
    grad += grad_pair
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

def _hex_init(rng, jitter=0.0):
    """Hexagonal packing initialization."""
    pts = []
    a = 0.11
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            y = -HALF + a + row * dy
            if y < HALF - a + 1e-9:
                pts.append([x, y])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    return rng.uniform(-HALF + 0.04, HALF - 0.04, size=(N, 2))

def _sa_escape(pts, r_target, deadline, rng, n_steps=3000):
    """Simulated annealing to escape local minima."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_viol = _violation(pts, r_target)
    current_viol = best_viol
    t_start = 0.008
    t_end = 0.00005
    for step in range(n_steps):
        if time.time() > deadline:
            break
        progress = step / max(n_steps - 1, 1)
        temp = t_start * (t_end / t_start) ** progress
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

def construct():
    start = time.time()
    deadline = start + 145.0
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def update_best(pts):
        nonlocal best_pts, best_r
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        return r
    
    # Phase 1: Multi-start with diverse initializations
    inits = [
        _hex_init(rng, jitter=0.003),
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.025),
        _grid_init(rng, jitter=0.008),
        _grid_init(rng, jitter=0.02),
        _random_init(rng),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 50.0:
            break
        # Soft-min polish with increasing beta (higher betas sharpen the min approximation)
        pts = _lbfgs_softmin(init, beta=100.0, deadline=deadline - 45.0)
        pts = _lbfgs_softmin(pts, beta=300.0, deadline=deadline - 40.0)
        pts = _lbfgs_softmin(pts, beta=600.0, deadline=deadline - 35.0)
        update_best(pts)
    
    if best_pts is None:
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Contact-graph local search to escape local minima
    r_current = best_r
    for _ in range(5):
        if time.time() > deadline - 25.0:
            break
        # Try to push radius slightly higher
        r_try = r_current + 0.0002
        # SA to find a configuration that might work at higher radius
        pts_sa, _ = _sa_escape(best_pts, r_try, deadline - 22.0, rng, n_steps=1500)
        # Contact graph local search
        pts_cg, _ = _contact_graph_local_search(pts_sa, r_try, deadline - 18.0, rng, n_iters=1200)
        # L-BFGS violation polish
        pts_v = _lbfgs_violation(pts_cg, r_try, deadline - 15.0)
        r_new = update_best(pts_v)
        if r_new > r_current + 1e-6:
            r_current = r_new
        else:
            # Try soft-min polish on current best with higher beta
            pts_sm = _lbfgs_softmin(best_pts, beta=500.0, deadline=deadline - 12.0)
            update_best(pts_sm)
    
    # Phase 3: High-precision soft-min polish (higher betas for sharper min approximation)
    if time.time() < deadline - 10.0:
        for beta in [800.0, 1200.0, 1600.0]:
            if time.time() > deadline - 6.0:
                break
            pts = _lbfgs_softmin(best_pts, beta=beta, deadline=deadline - 5.0)
            update_best(pts)
    
    # Phase 4: Final contact-graph refinement
    if time.time() < deadline - 5.0:
        pts_cg, _ = _contact_graph_local_search(best_pts, best_r, deadline - 3.0, rng, n_iters=400)
        update_best(pts_cg)
        pts_sm = _lbfgs_softmin(best_pts, beta=2000.0, deadline=deadline - 1.0)
        update_best(pts_sm)
    
    return best_pts
