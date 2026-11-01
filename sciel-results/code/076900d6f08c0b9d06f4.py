"""Packing 300 equal circles in the unit disk via efficient soft-min + force-directed polish."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

# EVOLVE-BLOCK-START
import time
import sys

def _hex_initial(n):
    """Generate n points in a hexagonal lattice inside the unit disk."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    best_points = None
    best_min_dist = 0.0
    for attempt in range(30):
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) >= n:
            rows.sort(key=lambda t: t[2])
            pts = np.array([(t[0], t[1]) for t in rows[:n]])
            d = cdist(pts, pts).min()
            if d > best_min_dist:
                best_min_dist = d
                best_points = pts
        spacing *= 0.97
    if best_points is None:
        pts = np.zeros((n, 2))
        pts[0] = [0, 0]
        for i in range(1, n):
            ring = int(np.ceil(np.sqrt(i)))
            angle = 2 * np.pi * i / max(1, ring)
            r = 0.95 * np.sqrt(i / n)
            pts[i] = [r * np.cos(angle), r * np.sin(angle)]
        return pts
    return best_points


def _ring_initial(n):
    """Concentric ring initialization with optimized ring radii."""
    pts = np.zeros((n, 2))
    pts[0] = [0, 0]
    remaining = n - 1
    num_rings = int(np.ceil(np.sqrt(remaining)))
    ring_counts = []
    per = remaining // num_rings
    extra = remaining % num_rings
    for i in range(num_rings):
        c = per + (1 if i < extra else 0)
        ring_counts.append(c)
    idx = 1
    for ri, count in enumerate(ring_counts):
        radius = 0.95 * np.sqrt((ri + 1) / (num_rings + 0.5))
        for j in range(count):
            if idx >= n:
                break
            angle = 2 * np.pi * j / count + 2 * np.pi * ri / (count + 1)
            pts[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return pts


def _random_initial(n, seed=42):
    """Random initialization with rejection sampling."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((n, 2))
    pts[0] = [0, 0]
    min_dist = 0.9 * 2.0 / (np.sqrt(n) + 1.0)
    for i in range(1, n):
        for attempt in range(200):
            r = np.sqrt(rng.rand()) * 0.95
            a = rng.rand() * 2 * np.pi
            x, y = r * np.cos(a), r * np.sin(a)
            if np.all(np.hypot(pts[:i, 0] - x, pts[:i, 1] - y) > min_dist):
                pts[i] = [x, y]
                break
        else:
            ring = int(np.ceil(np.sqrt(i)))
            angle = 2 * np.pi * i / max(1, ring)
            r = 0.9 * np.sqrt(i / n)
            pts[i] = [r * np.cos(angle), r * np.sin(angle)]
    return pts


def _true_score(x):
    """Compute the true score: min(min_pairwise/2, min_boundary)."""
    x = x.reshape(-1, 2)
    n = x.shape[0]
    dists = cdist(x, x)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    min_bound = (1.0 - radii).min()
    return min(min_pair, min_bound)


def _force_directed_objective(x, k_repulse, k_boundary):
    """
    Force-directed energy: repulsion between close pairs + boundary penalty.
    Much more efficient than soft-min for this problem.
    """
    x = x.reshape(-1, 2)
    n = x.shape[0]
    dists = cdist(x, x)
    np.fill_diagonal(dists, np.inf)
    
    # Repulsion: only consider pairs closer than 2*r_est (use current min as proxy)
    r_est = max(dists.min() / 2.0, 1e-6)
    cutoff = 2.5 * r_est
    
    energy = 0.0
    mask = dists < cutoff
    np.fill_diagonal(mask, False)
    if np.any(mask):
        # Inverse-square repulsion for close pairs
        close_dists = dists[mask]
        energy += np.sum(1.0 / (close_dists ** 2))
    
    # Boundary: penalty for being too close to edge
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    boundary_violation = np.maximum(0, radii - (1.0 - r_est))
    energy += k_boundary * np.sum(boundary_violation ** 2)
    
    return k_repulse * energy


def _force_directed_grad(x, k_repulse, k_boundary):
    """Gradient of force-directed energy."""
    x = x.reshape(-1, 2)
    n = x.shape[0]
    dists = cdist(x, x)
    np.fill_diagonal(dists, np.inf)
    
    r_est = max(dists.min() / 2.0, 1e-6)
    cutoff = 2.5 * r_est
    
    grad = np.zeros_like(x)
    mask = dists < cutoff
    np.fill_diagonal(mask, False)
    
    # Repulsion gradient: d(1/d^2)/dx_i = -2*(x_i - x_j)/d^4
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                d = dists[i, j]
                if d > 1e-10:
                    diff = x[i] - x[j]
                    grad[i] += k_repulse * (-2.0 / d ** 4) * diff
                    grad[j] += k_repulse * (2.0 / d ** 4) * diff
    
    # Boundary gradient
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    boundary_violation = np.maximum(0, radii - (1.0 - r_est))
    for i in range(n):
        if boundary_violation[i] > 0:
            r = radii[i]
            if r > 1e-10:
                grad[i] += 2.0 * k_boundary * boundary_violation[i] * x[i] / r
    
    return grad


def _soft_min_objective(x, beta):
    """Efficient soft-min surrogate using only nearest neighbors."""
    x = x.reshape(-1, 2)
    n = x.shape[0]
    dists = cdist(x, x)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    bound_vals = 1.0 - radii
    
    # Use top-k nearest pairs for efficiency
    k = 12
    all_vals = np.concatenate([pair_vals.ravel(), bound_vals])
    v_min = all_vals.min()
    if v_min < 0:
        return -1000.0 + beta * abs(v_min)
    
    # Only use the k smallest pair values + all boundary values
    flat_pairs = pair_vals.ravel()
    sorted_idx = np.argsort(flat_pairs)[:k]
    selected_pairs = flat_pairs[sorted_idx]
    selected_vals = np.concatenate([selected_pairs, bound_vals])
    
    neg_beta_v = -beta * selected_vals
    max_neg = neg_beta_v.max()
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_v - max_neg)))
    soft_min_val = -log_sum / beta
    return -soft_min_val


def _soft_min_grad(x, beta, k=12):
    """Efficient gradient using only k nearest pairs."""
    x = x.reshape(-1, 2)
    n = x.shape[0]
    dists = cdist(x, x)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    bound_vals = 1.0 - radii
    
    if all_vals_min := min(pair_vals.min(), bound_vals.min()) < 0:
        return np.zeros_like(x)
    
    flat_pairs = pair_vals.ravel()
    sorted_idx = np.argsort(flat_pairs)[:k]
    
    # Build weight vector for selected pairs + all bounds
    selected_pair_vals = flat_pairs[sorted_idx]
    all_selected = np.concatenate([selected_pair_vals, bound_vals])
    
    neg_beta_v = -beta * all_selected
    max_neg = neg_beta_v.max()
    exp_terms = np.exp(neg_beta_v - max_neg)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(x)
    
    # Boundary part
    bound_weights = weights[k:]
    for i in range(n):
        r = radii[i]
        if r > 1e-10:
            grad[i] += bound_weights[i] * x[i] / r
    
    # Pairwise part for k nearest
    for idx in range(k):
        i = idx // n
        j = idx % n
        if i == j:
            continue
        w = weights[idx]
        d = dists[i, j]
        if d > 1e-10:
            diff = x[i] - x[j]
            unit = diff / d
            grad[i] += w * unit
            grad[j] -= w * unit
    
    return -grad


def _active_set_polish(x, max_iter=200, tol=1e-15):
    """
    Efficient active-set polish: iteratively identify binding constraints
    and solve a small QP to maximize r.
    """
    x = x.copy().reshape(-1, 2)
    n = x.shape[0]
    
    for it in range(max_iter):
        dists = cdist(x, x)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        radii = np.sqrt(np.sum(x ** 2, axis=1))
        bound_vals = 1.0 - radii
        r = min(pair_vals.min(), bound_vals.min())
        
        # Find active constraints
        tol_active = max(1e-12, 1e-8 * r)
        active_pair_mask = pair_vals <= r + tol_active
        active_bound_mask = bound_vals <= r + tol_active
        
        n_active_pairs = np.sum(active_pair_mask) // 2  # symmetric
        n_active_bounds = np.sum(active_bound_mask)
        
        if n_active_pairs == 0 and n_active_bounds == 0:
            break
        
        # Compute gradient of r (direction to increase minimum constraint)
        grad_r = np.zeros_like(x)
        
        # Boundary contributions
        active_bound_idx = np.where(active_bound_mask)[0]
        for i in active_bound_idx:
            ri = radii[i]
            if ri > 1e-10:
                grad_r[i] += x[i] / ri
        
        # Pairwise contributions (only active pairs)
        for i in range(n):
            for j in range(i + 1, n):
                if active_pair_mask[i, j]:
                    d_ij = dists[i, j]
                    if d_ij < 1e-10:
                        continue
                    vec = (x[i] - x[j]) / (2.0 * d_ij)
                    grad_r[i] += vec
                    grad_r[j] -= vec
        
        gn = np.linalg.norm(grad_r)
        if gn < 1e-15:
            break
        
        # Adaptive step size with backtracking
        best_x_trial = None
        best_r_trial = r
        alpha = 1e-6
        for _ in range(20):
            x_trial = x + alpha * grad_r / gn
            dists_t = cdist(x_trial, x_trial)
            np.fill_diagonal(dists_t, np.inf)
            r_trial = min(dists_t.min() / 2.0, (1.0 - np.sqrt(np.sum(x_trial ** 2, axis=1))).min())
            if r_trial > best_r_trial + 1e-15:
                best_r_trial = r_trial
                best_x_trial = x_trial.copy()
            alpha *= 0.5
            if alpha < 1e-14:
                break
        
        if best_x_trial is None:
            break
        
        x = best_x_trial
        r = best_r_trial
    
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin for 150s limit
    
    # Try multiple initializations
    inits = [_hex_initial(N), _ring_initial(N), _random_initial(N, seed=42), 
             _random_initial(N, seed=123), _random_initial(N, seed=999)]
    
    best_x = None
    best_r = 0.0
    for init in inits:
        r_init = _true_score(init)
        if np.isfinite(r_init) and r_init > best_r:
            best_r = r_init
            best_x = init.copy()
    
    x0_flat = best_x.ravel()
    
    # Phase 1: Force-directed optimization (fast, good for coarse packing)
    if time.time() < deadline - 60.0:
        try:
            res = minimize(
                _force_directed_objective,
                x0_flat,
                args=(1.0, 50.0),
                method='L-BFGS-B',
                jac=_force_directed_grad,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 2000}
            )
            x_opt = res.x.reshape(-1, 2)
            r_opt = _true_score(x_opt)
            if np.isfinite(r_opt) and r_opt > best_r:
                best_r = r_opt
                best_x = x_opt.copy()
                x0_flat = x_opt.ravel()
        except Exception:
            pass
    
    # Phase 2: Soft-min annealing with efficient k-nearest evaluation
    betas = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    
    for beta in betas:
        if time.time() > deadline - 30.0:
            break
        
        try:
            opts = {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1500}
            
            res = minimize(
                _soft_min_objective,
                x0_flat,
                args=(beta,),
                method='L-BFGS-B',
                jac=lambda x: _soft_min_grad(x, beta),
                options=opts
            )
            x_opt = res.x.reshape(-1, 2)
            r_opt = _true_score(x_opt)
            
            if np.isfinite(r_opt) and r_opt > best_r:
                best_r = r_opt
                best_x = x_opt.copy()
                x0_flat = x_opt.ravel()
        except Exception:
            pass
    
    # Phase 3: Active-set polish for final precision
    if time.time() < deadline - 5.0:
        try:
            x_polished = _active_set_polish(best_x, max_iter=150, tol=1e-15)
            r_polished = _true_score(x_polished)
            if np.isfinite(r_polished) and r_polished > best_r:
                best_r = r_polished
                best_x = x_polished.copy()
        except Exception:
            pass
    
    # Final validation
    x_final = best_x.copy()
    radii = np.sqrt(np.sum(x_final ** 2, axis=1))
    for i in range(len(x_final)):
        if radii[i] > 1.0:
            x_final[i] = x_final[i] / radii[i] * 0.999
    
    return x_final
# EVOLVE-BLOCK-END
