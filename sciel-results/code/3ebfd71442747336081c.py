"""Packing of 50 equal circles in a unit square via contact-graph optimization + SA + L-BFGS polish."""
import numpy as np
import time

N = 50
HALF = 0.5

try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False


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
    if _HAS_KDTREE:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(1.0)
        if pairs:
            idx_i = [p[0] for p in pairs]
            idx_j = [p[1] for p in pairs]
            diffs = pts[idx_i] - pts[idx_j]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            r = min(r, float(np.min(dists)) / 2.0)
    else:
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r = min(r, float(np.min(dist)) / 2.0)
    return r


def _soft_min_radius(pts, beta=50.0):
    """Smooth soft-minimum surrogate for the radius using log-sum-exp.
    
    r_soft = -1/beta * log(sum(exp(-beta * d_i))) where d_i are all constraints
    This is differentiable and approximates min(d_i) as beta -> infinity.
    """
    # Wall distances
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    # Pairwise distances / 2
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    # Combine all constraints
    all_d = np.concatenate([d_left, d_right, d_bottom, d_top, pair_dists])
    
    # Soft min via log-sum-exp for numerical stability
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = max_d - log_sum / beta
    
    return float(r_soft)


def _soft_min_radius_grad(pts, beta=50.0):
    """Gradient of the soft-minimum radius surrogate."""
    # Wall distances
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    # Pairwise distances / 2
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    # Combine all constraints
    all_d = np.concatenate([d_left, d_right, d_bottom, d_top, pair_dists])
    
    # Soft min gradient
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Wall gradients: d(d_left)/d(x_i) = 1, d(d_right)/d(x_i) = -1, etc.
    # d_left[i] = x_i + HALF, so d/dx_i = 1
    grad[:, 0] += weights[:N]  # d_left
    grad[:, 0] -= weights[N:2*N]  # d_right
    grad[:, 1] += weights[2*N:3*N]  # d_bottom
    grad[:, 1] -= weights[3*N:4*N]  # d_top
    
    # Pairwise gradients: d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        w = weights[4*N + k]
        if w > 1e-15:
            dij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(dij)
            if dist_ij > 1e-15:
                grad[i] += w * dij / (2.0 * dist_ij)
                grad[j] -= w * dij / (2.0 * dist_ij)
    
    return grad


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations."""
    if _HAS_KDTREE:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0 * r_target + 0.1)
        pair_viol_sum = 0.0
        if pairs:
            idx_i = [p[0] for p in pairs]
            idx_j = [p[1] for p in pairs]
            diffs = pts[idx_i] - pts[idx_j]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            viol = np.maximum(0.0, 2.0 * r_target - dists)
            pair_viol_sum = float(np.sum(viol))
    else:
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0.0, 2.0 * r_target - dist)
        iu = np.triu_indices(N, k=1)
        pair_viol = pair_viol[iu]
        pair_viol_sum = float(np.sum(pair_viol))
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return pair_viol_sum + float(np.sum(wall_viol))


def _violation_grad(pts, r_target):
    """Analytic gradient of the violation function."""
    grad = np.zeros((N, 2))
    
    # Wall gradient
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    for i in range(N):
        if d_left[i] < r_target:
            grad[i, 0] -= 1.0
        if d_right[i] < r_target:
            grad[i, 0] += 1.0
        if d_bottom[i] < r_target:
            grad[i, 1] -= 1.0
        if d_top[i] < r_target:
            grad[i, 1] += 1.0
    
    # Pairwise gradient
    if _HAS_KDTREE:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0 * r_target + 0.1)
        if pairs:
            for i, j in pairs:
                dij = pts[i] - pts[j]
                dist_ij = np.linalg.norm(dij)
                if dist_ij < 2.0 * r_target and dist_ij > 1e-15:
                    unit = dij / dist_ij
                    grad[i] += unit
                    grad[j] -= unit
    else:
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        iu = np.triu_indices(N, k=1)
        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            dist_ij = dist[i, j]
            if dist_ij < 2.0 * r_target and dist_ij > 1e-15:
                dij = pts[i] - pts[j]
                unit = dij / dist_ij
                grad[i] += unit
                grad[j] -= unit
    
    return grad


def _contact_graph(pts, r_target, tolerance=1e-6):
    """Identify contact pairs: pairs where distance is within 2*r_target of contact."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Contact if dist is close to 2*r_target (within tolerance)
    contacts = np.abs(dist - 2.0 * r_target) < tolerance
    iu = np.triu_indices(N, k=1)
    contact_pairs = [(iu[0][k], iu[1][k]) for k in range(len(iu[0])) if contacts[iu[0][k], iu[1][k]]]
    return contact_pairs


def _wall_contacts(pts, r_target, tolerance=1e-6):
    """Identify wall contacts: points where distance to wall is close to r_target."""
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_contacts = []
    for i in range(N):
        if abs(d_left[i] - r_target) < tolerance:
            wall_contacts.append((i, 'left'))
        if abs(d_right[i] - r_target) < tolerance:
            wall_contacts.append((i, 'right'))
        if abs(d_bottom[i] - r_target) < tolerance:
            wall_contacts.append((i, 'bottom'))
        if abs(d_top[i] - r_target) < tolerance:
            wall_contacts.append((i, 'top'))
    return wall_contacts


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=3000):
    """Simulated annealing to minimize violation for target radius r_target."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    # Adaptive temperature
    t_start = 0.03
    t_end = 0.00001
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    no_improve = 0
    step_scale = 1.0
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Adaptive step size
        if no_improve > 200:
            step_scale *= 1.1
            no_improve = 0
        elif no_improve > 500:
            step_scale *= 0.8
            no_improve = 0
        
        # Random move: pick 1-3 circles and perturb
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp * step_scale, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    return best_pts, best_viol


def _lbfgs_violation(pts0, r_target, deadline):
    """Use L-BFGS-B to minimize violation for target radius with analytic gradient."""
    from scipy.optimize import minimize
    
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
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(res.fun)


def _lbfgs_soft_min(pts0, deadline, beta_start=30.0, beta_end=300.0, n_iters=6):
    """Use L-BFGS-B to maximize the soft-minimum radius surrogate.
    
    This is a smooth objective that approximates the true min radius.
    We increase beta over iterations to sharpen the approximation.
    """
    from scipy.optimize import minimize
    
    def neg_obj(x, beta):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_soft_min_radius(p, beta)
    
    def neg_grad(x, beta):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_soft_min_radius_grad(p, beta).ravel()
    
    x = pts0.ravel().copy()
    best_x = x.copy()
    best_val = _soft_min_radius(pts0, beta_start)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iters - 1, 1))
        
        try:
            res = minimize(neg_obj, x, method='L-BFGS-B',
                          jac=lambda x: neg_grad(x, beta),
                          bounds=[(-HALF, HALF)] * (2 * N),
                          options={'maxiter': 300, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 50})
            x_new = res.x
            x_new = np.clip(x_new, -HALF, HALF)
            val = _soft_min_radius(x_new.reshape(N, 2), beta)
            if val > best_val:
                best_val = val
                best_x = x_new.copy()
            x = x_new
        except Exception:
            break
    
    pts = best_x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.118
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
    """Grid initialization with slight perturbation for better packing."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Add center point and adjust
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _binary_search_radius(pts_start, deadline, rng):
    """Binary search on radius with violation minimization."""
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    r_lo = best_r
    r_hi = 0.07145
    
    for it in range(12):
        if time.time() > deadline - 3.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to find feasible configuration at r_mid
        remaining = deadline - time.time() - 2.0
        n_sa = min(3000, max(500, int(remaining * 200)))
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_mid, deadline - 2.0, rng, n_steps=n_sa)
        
        # Polish with L-BFGS
        try:
            pts_lbfgs, viol_lbfgs = _lbfgs_violation(pts_sa, r_mid, deadline - 1.0)
        except Exception:
            pts_lbfgs = pts_sa
            viol_lbfgs = viol_sa
        
        # Check if feasible
        r_achieved = _compute_r(pts_lbfgs)
        
        if viol_lbfgs < 1e-8 and r_achieved >= r_mid - 1e-10:
            r_lo = r_mid
            best_pts = pts_lbfgs.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s limit
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps):
        nonlocal best_pts, best_r
        if time.time() > deadline - 8.0:
            return
        
        remaining = deadline - time.time() - 3.0
        n_sa = min(n_sa_steps, max(300, int(remaining * 150)))
        
        # SA phase
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - 3.0, rng, n_steps=n_sa)
        
        # L-BFGS polish with analytic gradient (fast)
        try:
            pts_lbfgs, viol_lbfgs = _lbfgs_violation(pts_sa, r_target, deadline - 1.0)
        except Exception:
            pts_lbfgs = pts_sa
            viol_lbfgs = viol_sa
        
        r = _compute_r(pts_lbfgs)
        if r > best_r:
            best_r = r
            best_pts = pts_lbfgs.copy()
    
    # Phase 1: Multi-start with moderate target
    inits = [
        _hex_init(rng, jitter=0.005),
        _hex_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.01),
        _grid_init(rng, jitter=0.04),
        _random_init(rng),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        try_improve(init, 0.0705, 2500)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search on radius
    if time.time() < deadline - 30.0:
        best_pts, best_r = _binary_search_radius(best_pts, deadline, rng)
    
    # Phase 3: Soft-min L-BFGS polish for high precision
    if time.time() < deadline - 12.0:
        try:
            pts_soft = _lbfgs_soft_min(best_pts, deadline - 6.0, 
                                        beta_start=30.0, beta_end=300.0, n_iters=6)
            r_soft = _compute_r(pts_soft)
            if r_soft > best_r:
                best_r = r_soft
                best_pts = pts_soft
        except Exception:
            pass
    
    # Phase 4: Final violation-based polish with very tight target
    if time.time() < deadline - 6.0:
        r_target = best_r * 1.0003
        try_improve(best_pts, r_target, 4000)
    
    # Phase 5: Very final soft-min polish with high beta
    if time.time() < deadline - 3.0:
        try:
            pts_soft = _lbfgs_soft_min(best_pts, deadline - 1.0,
                                        beta_start=80.0, beta_end=800.0, n_iters=4)
            r_soft = _compute_r(pts_soft)
            if r_soft > best_r:
                best_r = r_soft
                best_pts = pts_soft
        except Exception:
            pass
    
    # Phase 6: Ultra-final L-BFGS violation polish
    if time.time() < deadline - 1.0:
        try:
            r_target = best_r * 1.0001
            pts_final, viol_final = _lbfgs_violation(best_pts, r_target, deadline - 0.1)
            r_final = _compute_r(pts_final)
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final
        except Exception:
            pass
    
    return best_pts
