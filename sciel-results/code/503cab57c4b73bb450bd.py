"""Packing of 50 equal circles in a unit square via contact-graph Gauss-Seidel + SA + L-BFGS polish."""
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


def _build_contact_graph(pts, r_target, tol=1e-6):
    """
    Build a contact graph where edges represent pairs of circles that are
    close to touching (within tolerance) and wall contacts.
    Returns:
        contacts: list of (i, j) pairs
        wall_contacts: list of (i, wall_id) where wall_id: 0=left,1=right,2=bottom,3=top
    """
    contacts = []
    wall_contacts = []
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i, j] < 2.0 * r_target + tol:
                contacts.append((i, j))
    
    # Wall contacts
    for i in range(N):
        x, y = pts[i]
        if abs(x + HALF - r_target) < tol:
            wall_contacts.append((i, 0))
        if abs(HALF - x - r_target) < tol:
            wall_contacts.append((i, 1))
        if abs(y + HALF - r_target) < tol:
            wall_contacts.append((i, 2))
        if abs(HALF - y - r_target) < tol:
            wall_contacts.append((i, 3))
    
    return contacts, wall_contacts


def _gauss_seidel_solve(pts, r_target, max_iter=10000, tol=1e-12, deadline=None):
    """
    Gauss-Seidel iterative solver for the contact graph.
    Each iteration updates each circle's position to satisfy its contact constraints.
    """
    pts = pts.copy()
    contacts, wall_contacts = _build_contact_graph(pts, r_target, tol=1e-5)
    
    # Precompute contact neighbor lists
    neighbors = [[] for _ in range(N)]
    for i, j in contacts:
        neighbors[i].append(j)
        neighbors[j].append(i)
    
    # Precompute wall contact info
    wall_info = {}  # i -> list of (wall_id, sign_x, sign_y)
    for i, wid in wall_contacts:
        if i not in wall_info:
            wall_info[i] = []
        if wid == 0:  # left wall: x = -HALF + r
            wall_info[i].append((0, 1, 0))
        elif wid == 1:  # right wall: x = HALF - r
            wall_info[i].append((1, -1, 0))
        elif wid == 2:  # bottom wall: y = -HALF + r
            wall_info[i].append((2, 0, 1))
        elif wid == 3:  # top wall: y = HALF - r
            wall_info[i].append((3, 0, -1))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        max_change = 0.0
        
        for i in range(N):
            x, y = pts[i]
            new_x, new_y = x, y
            
            # Update based on wall contacts
            if i in wall_info:
                for wid, sx, sy in wall_info[i]:
                    if wid == 0:  # left wall
                        new_x = -HALF + r_target
                    elif wid == 1:  # right wall
                        new_x = HALF - r_target
                    elif wid == 2:  # bottom wall
                        new_y = -HALF + r_target
                    elif wid == 3:  # top wall
                        new_y = HALF - r_target
            
            # Update based on circle contacts
            for j in neighbors[i]:
                dx = x - pts[j, 0]
                dy = y - pts[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-15:
                    continue
                # Move i away from j to maintain distance 2*r
                target_dist = 2.0 * r_target
                if dist < target_dist:
                    # Push apart
                    factor = target_dist / dist
                    new_x = pts[j, 0] + factor * dx
                    new_y = pts[j, 1] + factor * dy
            
            # Clamp to square
            new_x = np.clip(new_x, -HALF, HALF)
            new_y = np.clip(new_y, -HALF, HALF)
            
            change = abs(new_x - x) + abs(new_y - y)
            max_change = max(max_change, change)
            pts[i, 0] = new_x
            pts[i, 1] = new_y
        
        if max_change < tol:
            break
    
    return pts


def _contact_graph_optimize(pts, r_target, deadline, max_gs_iter=5000, n_restarts=5):
    """
    Optimize using contact graph Gauss-Seidel with multiple restarts.
    """
    best_pts = pts.copy()
    best_r = _compute_r(pts)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Add small perturbation for restart
        if restart > 0:
            perturbation = 0.001 * (restart + 1)
            rng = np.random.default_rng(42 + restart)
            pts_pert = pts + rng.normal(0, perturbation, size=pts.shape)
            pts_pert = np.clip(pts_pert, -HALF, HALF)
        else:
            pts_pert = pts.copy()
        
        # Run Gauss-Seidel
        pts_gs = _gauss_seidel_solve(pts_pert, r_target, max_iter=max_gs_iter, deadline=deadline - 5.0)
        
        # Check result
        r_achieved = _compute_r(pts_gs)
        if r_achieved > best_r:
            best_r = r_achieved
            best_pts = pts_gs.copy()
        
        # If we achieved the target, try to increase it slightly
        if r_achieved >= r_target - 1e-9:
            # Try a slightly larger radius
            r_new = r_target * 1.0001
            pts_try = _gauss_seidel_solve(best_pts, r_new, max_iter=max_gs_iter // 2, deadline=deadline - 5.0)
            r_new_achieved = _compute_r(pts_try)
            if r_new_achieved > best_r:
                best_r = r_new_achieved
                best_pts = pts_try.copy()
    
    return best_pts, best_r


def _soft_min_obj(pts, r_target, beta=500.0):
    """
    Smooth surrogate for the packing problem.
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
    Gradient of the soft-min objective with respect to pts.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    
    pair_viol = 2.0 * r_target - dist
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-beta * pair_viol_clipped))
    
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d < 1e-12:
                continue
            s = sig[i, j]
            diff_ij = (pts[i] - pts[j]) / d
            grad[i] -= s * diff_ij
            grad[j] += s * diff_ij
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    sig_w = 1.0 / (1.0 + np.exp(-beta * wall_viol_clipped))
    
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = sig_w[i]
        if abs(dl - wmin) < 1e-10:
            grad[i, 0] -= s
        if abs(dr - wmin) < 1e-10:
            grad[i, 0] += s
        if abs(db - wmin) < 1e-10:
            grad[i, 1] -= s
        if abs(dt - wmin) < 1e-10:
            grad[i, 1] += s
    
    return grad


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist)
    iu = np.triu_indices(N, k=1)
    pair_viol = pair_viol[iu]
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=3000):
    """Simulated annealing to minimize violation for target radius r_target."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    # Temperature schedule
    t_start = 0.04
    t_end = 0.00005
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Random move: pick 1-3 circles and perturb
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
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
                   options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-10})
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
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _binary_search_radius(pts_start, deadline, rng):
    """Binary search on radius with soft-min L-BFGS minimization."""
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    r_lo = best_r
    r_hi = 0.07138
    
    for _ in range(12):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to find feasible configuration at r_mid
        # First SA to get into basin
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_mid, deadline - 5.0, rng, n_steps=1500)
        
        # Polish with L-BFGS on soft-min objective
        try:
            pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_mid, deadline - 5.0, beta=500.0, maxiter=300)
        except Exception:
            pts_lbfgs = pts_sa
            obj_lbfgs = _soft_min_obj(pts_sa, r_mid, 500.0)
        
        # Check if feasible
        r_achieved = _compute_r(pts_lbfgs)
        viol = _violation(pts_lbfgs, r_mid)
        
        if viol < 1e-8 and r_achieved >= r_mid - 1e-9:
            r_lo = r_mid
            best_pts = pts_lbfgs.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def _sdp_relaxation_init(rng):
    """
    Attempt an SDP-inspired initialization by solving a relaxed version
    where we maximize the minimum pairwise distance using a smooth
    log-sum-exp surrogate, then use that as a starting point.
    """
    from scipy.optimize import minimize
    
    # Objective: minimize -log-sum-exp of pairwise distances (smooth max)
    # This encourages spreading points apart
    def obj(x):
        p = x.reshape(N, 2)
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 0.0)
        dist = np.sqrt(dist_sq)
        iu = np.triu_indices(N, k=1)
        d = dist[iu]
        # log-sum-exp for numerical stability
        max_d = np.max(d)
        lse = max_d + np.log(np.sum(np.exp(d - max_d)))
        # Also penalize wall violations softly
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
        weights = exp_d / sum_exp  # softmax weights
        
        g = np.zeros_like(p)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[idx]
            diff_ij = p[i] - p[j]
            d_ij = dist_safe[i, j]
            g[i] += w * diff_ij / d_ij
            g[j] -= w * diff_ij / d_ij
        
        # Wall gradient
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
    
    # Start from a hexagonal lattice
    x0 = _hex_init(rng, jitter=0.02).ravel()
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return _hex_init(rng, jitter=0.02)


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s limit
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    # Phase 0: Initialize with hexagonal lattice
    init_pts = _hex_init(rng, jitter=0.01)
    best_pts = init_pts.copy()
    best_r = _compute_r(best_pts)
    
    # Phase 1: Contact graph optimization with increasing radius targets
    # Start with a moderate target and gradually increase
    r_targets = [0.065, 0.068, 0.070, 0.071, 0.0713]
    
    for r_target in r_targets:
        if time.time() > deadline - 20.0:
            break
        
        # Try contact graph optimization
        pts_cg, r_cg = _contact_graph_optimize(best_pts, r_target, deadline - 20.0, 
                                                max_gs_iter=10000, n_restarts=3)
        if r_cg > best_r:
            best_r = r_cg
            best_pts = pts_cg.copy()
        
        # If contact graph didn't reach target, try SA + L-BFGS as fallback
        if r_cg < r_target - 1e-6 and time.time() < deadline - 15.0:
            pts_sa, viol_sa = _simulated_annealing(best_pts, r_target, deadline - 15.0, rng, n_steps=2000)
            try:
                pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_target, deadline - 15.0, beta=500.0, maxiter=300)
            except Exception:
                pts_lbfgs = pts_sa
            r_lbfgs = _compute_r(pts_lbfgs)
            if r_lbfgs > best_r:
                best_r = r_lbfgs
                best_pts = pts_lbfgs.copy()
    
    # Phase 2: Fine-tune with binary search near the best found radius
    if time.time() < deadline - 10.0:
        r_lo = best_r
        r_hi = 0.07138
        
        for _ in range(8):
            if time.time() > deadline - 5.0:
                break
            
            r_mid = (r_lo + r_hi) / 2.0
            
            # Try to achieve r_mid with contact graph
            pts_try, r_achieved = _contact_graph_optimize(best_pts, r_mid, deadline - 5.0,
                                                           max_gs_iter=5000, n_restarts=2)
            
            if r_achieved >= r_mid - 1e-8:
                r_lo = r_mid
                if r_achieved > best_r:
                    best_r = r_achieved
                    best_pts = pts_try.copy()
            else:
                r_hi = r_mid
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3.0:
        # Use L-BFGS with very tight tolerance for final polish
        r_target = best_r * 1.00001
        try:
            pts_final, _ = _lbfgs_polish(best_pts, r_target, deadline - 3.0, beta=2000.0, maxiter=500)
            r_final = _compute_r(pts_final)
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final.copy()
        except Exception:
            pass
        
        # One more contact graph pass at the best radius
        pts_cg_final, r_cg_final = _contact_graph_optimize(best_pts, best_r, deadline - 3.0,
                                                            max_gs_iter=20000, n_restarts=5)
        if r_cg_final > best_r:
            best_r = r_cg_final
            best_pts = pts_cg_final.copy()
    
    return best_pts
