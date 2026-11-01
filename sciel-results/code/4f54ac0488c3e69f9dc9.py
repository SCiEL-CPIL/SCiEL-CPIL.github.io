"""Circle packing via radial search + soft-min L-BFGS-B polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    if n < 2:
        return 0.0
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)


def _softmin(vals, beta):
    """Log-sum-exp approximation of min(vals)."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return 0.0
    v_max = np.max(vals)
    log_sum = np.log(np.sum(np.exp(beta * (vals - v_max)))) / beta
    return v_max + log_sum


def _objective_and_grad(x, beta, target_r):
    """
    Objective: minimize violation of constraints.
    - Pair distances should be >= 2*r_target
    - Each center should be within (1 - r_target) of origin
    
    Returns (objective, gradient)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    r = target_r
    
    # Pair distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Container margins
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    
    # Violations: we want dist >= 2r and radii <= 1-r
    # Define gap_i = dist_i - 2r (should be >= 0)
    # Define margin_i = 1 - r - radii_i (should be >= 0)
    gaps = dists - 2 * r
    margins = (1 - r) - radii
    
    # Combine into softmin
    all_vals = np.concatenate([gaps.flatten(), margins.flatten()])
    # Filter out diagonal (which is inf)
    all_vals = all_vals[all_vals < 1e15]
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft-min of all constraints
    obj = -_softmin(all_vals, beta)  # minimize negative softmin = maximize softmin
    
    # Gradient
    grad = np.zeros_like(x)
    
    # For pair gaps: d/dx_i (d_ij - 2r) = (x_i - x_j) / d_ij
    # weight for softmin: w_k = exp(beta * v_k) / sum(exp(beta * v_j))
    # grad_obj = -sum_k w_k * grad_v_k
    
    # Pair contributions
    beta_exp = beta * (gaps - np.max(gaps))
    w_pairs = np.exp(beta_exp)
    w_pairs_sum = np.sum(w_pairs)
    if w_pairs_sum > 0:
        w_pairs = w_pairs / w_pairs_sum
    else:
        w_pairs = np.full_like(w_pairs, 1.0 / len(gaps[gaps < 1e15]))
    
    # Gradient of gap_ij w.r.t. x_i is (x_i - x_j)/d_ij
    # Sum over j for each i
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            w = w_pairs[i, j]
            grad[i] += -w * (pts[i] - pts[j]) / d
    
    # Margin contributions: d/dx_i (1 - r - |x_i|) = -x_i / |x_i|
    max_margin = np.max(margins)
    beta_exp_m = beta * (margins - max_margin)
    w_margins = np.exp(beta_exp_m)
    w_margins_sum = np.sum(w_margins)
    if w_margins_sum > 0:
        w_margins = w_margins / w_margins_sum
    else:
        w_margins = np.full_like(w_margins, 1.0 / n)
    
    for i in range(n):
        r_i = radii[i]
        if r_i < 1e-12:
            continue
        grad[i] += -w_margins[i] * (-pts[i] / r_i)
    
    # Combine: obj = -softmin(all), so grad_obj = -sum w * grad_v
    # But we defined w over all vals concatenated... let me redo more carefully.
    
    # Actually let's recompute properly
    grad = np.zeros_like(x)
    
    # Build list of all constraint values and their gradients
    # Constraint k for pair (i,j): v_k = dist_ij - 2r, grad w.r.t. x_i = (x_i - x_j)/d, grad w.r.t. x_j = (x_j - x_i)/d
    # Constraint for center i: v_k = 1 - r - |x_i|, grad w.r.t. x_i = -x_i/|x_i|
    
    all_vals_list = []
    grad_list = []  # each element is a vector of length 2n
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 1e-12:
                continue
            all_vals_list.append(d - 2 * r)
            g = np.zeros(2 * n)
            g[i * 2] = (pts[i][0] - pts[j][0]) / d
            g[i * 2 + 1] = (pts[i][1] - pts[j][1]) / d
            g[j * 2] = (pts[j][0] - pts[i][0]) / d
            g[j * 2 + 1] = (pts[j][1] - pts[i][1]) / d
            grad_list.append(g)
    
    for i in range(n):
        r_i = radii[i]
        all_vals_list.append(1 - r - r_i)
        g = np.zeros(2 * n)
        if r_i > 1e-12:
            g[i * 2] = -pts[i][0] / r_i
            g[i * 2 + 1] = -pts[i][1] / r_i
        grad_list.append(g)
    
    if not all_vals_list:
        return 0.0, np.zeros_like(x)
    
    all_vals_arr = np.array(all_vals_list)
    max_v = np.max(all_vals_arr)
    exps = np.exp(beta * (all_vals_arr - max_v))
    weights = exps / np.sum(exps)
    
    obj = -(max_v + np.log(np.sum(exps)) / beta)
    
    grad_total = np.zeros(2 * n)
    for w, g in zip(weights, grad_list):
        grad_total += -w * g  # because obj = -softmin, grad_obj = -sum w * grad_v
    
    return obj, grad_total


def _optimize_for_radius(init_pts, target_r, time_limit, beta_start=50.0, beta_end=500.0):
    """
    Optimize positions to achieve minimum radius >= target_r.
    Returns (pts, success, score)
    """
    t0 = time.time()
    x0 = init_pts.flatten().astype(np.float64)
    
    # Try increasing beta for sharper min approximation
    best_score = _compute_score(init_pts)
    best_pts = init_pts.copy()
    
    num_betas = 5
    for bi in range(num_betas):
        if time.time() - t0 > time_limit:
            break
        if bi == 0:
            beta = beta_start
        else:
            beta = beta_start * (beta_end / beta_start) ** (bi / (num_betas - 1))
        
        def fun(x):
            if time.time() - t0 > time_limit:
                return 1e10, np.zeros_like(x)
            return _objective_and_grad(x, beta, target_r)
        
        try:
            res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15,
                                    'maxls': 50})
            x_opt = res.x
            pts_opt = x_opt.reshape(-1, 2)
            score_opt = _compute_score(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
            x0 = x_opt
        except Exception:
            pass
    
    return best_pts, best_score >= target_r - 1e-9, best_score


def _hex_init(spacing):
    """Hexagonal lattice initialization."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(radius, kind='stable')[:N]]
    return points


def _ring_init(n_rings=8):
    """Concentric rings initialization."""
    pts = [np.array([0.0, 0.0])]
    for ring in range(1, n_rings):
        r = ring * 0.12
        n_in_ring = min(6 * ring, N - len(pts))
        if n_in_ring <= 0:
            break
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        for a in angles:
            pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    pts = np.array(pts[:N])
    return pts


def _random_init(seed):
    """Random initialization within unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        x = rng.uniform(-1, 1, size=(N * 2, 2))
        r = np.hypot(x[:, 0], x[:, 1])
        valid = r <= 0.95
        pts.extend(x[valid].tolist())
    pts = np.array(pts[:N])
    return pts


def construct():
    t_start = time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Multiple initializations
    inits = [
        _hex_init(0.25),
        _hex_init(0.22),
        _hex_init(0.28),
        _ring_init(8),
        _ring_init(7),
        _random_init(42),
        _random_init(123),
        _random_init(7),
    ]
    
    # First pass: get a decent starting point via quick optimization at moderate radius
    r_initial = 0.115
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() - t_start > DEADLINE - 30:
            break
        if len(init_pts) < N:
            continue
        
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_opt = min(time_remaining * 0.4, 15.0)
        if time_for_opt < 2.0:
            break
        
        pts_opt, success, score = _optimize_for_radius(init_pts, r_initial, time_for_opt,
                                                       beta_start=20.0, beta_end=100.0)
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    if best_pts is None:
        best_pts = _hex_init(0.25)
        best_score = _compute_score(best_pts)
    
    # Binary search on radius to find the maximum feasible radius
    r_low = best_score
    r_high = 0.135  # slightly above known best
    
    # Do binary search
    num_bs_steps = 8
    for step in range(num_bs_steps):
        if time.time() - t_start > DEADLINE - 10:
            break
        r_mid = (r_low + r_high) / 2.0
        
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_opt = min(time_remaining * 0.5, 12.0)
        if time_for_opt < 1.0:
            break
        
        # Try to achieve radius r_mid
        # Use best current points as initial guess
        pts_feas, success, score = _optimize_for_radius(best_pts, r_mid, time_for_opt,
                                                        beta_start=50.0, beta_end=500.0)
        
        if success and score >= r_mid - 1e-9:
            r_low = r_mid
            if score > best_score:
                best_score = score
                best_pts = pts_feas.copy()
        else:
            r_high = r_mid
            # Update best with the best found anyway
            if score > best_score:
                best_score = score
                best_pts = pts_feas.copy()
    
    # Final high-precision polish
    if time.time() - t_start < DEADLINE - 5:
        time_limit = DEADLINE - (time.time() - t_start) - 2
        if time_limit > 1.0:
            # Use very high beta for final polish on current best
            final_beta = 1000.0
            
            def fun_final(x):
                if time.time() - t_start > DEADLINE:
                    return 1e10, np.zeros_like(x)
                return _objective_and_grad(x, final_beta, best_score)
            
            try:
                res = minimize(fun_final, best_pts.flatten(), jac=True, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15,
                                        'maxls': 50})
                pts_final = res.x.reshape(-1, 2)
                score_final = _compute_score(pts_final)
                if score_final > best_score:
                    best_score = score_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    return best_pts
