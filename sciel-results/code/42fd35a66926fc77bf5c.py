"""Packing 300 equal circles in the unit disk via soft-min surrogate + L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
import time
import sys
from scipy.spatial.distance import pdist
from scipy.linalg import solve

def _hex_initial(n):
    """Generate n points in a hexagonal lattice inside the unit disk,
    choosing spacing so we get at least n points, then pick the n closest to center."""
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
            d = pdist(pts)
            md = d.min()
            if md > best_min_dist:
                best_min_dist = md
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
    # Use golden-ratio-like spacing for rings
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
    """Random initialization with rejection sampling for minimum distance."""
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
            # Fallback: place on a ring
            ring = int(np.ceil(np.sqrt(i)))
            angle = 2 * np.pi * i / max(1, ring)
            r = 0.9 * np.sqrt(i / n)
            pts[i] = [r * np.cos(angle), r * np.sin(angle)]
    return pts


def _soft_min_objective(x, beta):
    """
    Soft-min surrogate for max-min packing.
    """
    x = x.reshape(-1, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    bound_vals = 1.0 - radii
    all_vals = np.concatenate([pair_vals.ravel(), bound_vals])
    v_min = all_vals.min()
    if v_min < 0:
        return -1000.0 + beta * abs(v_min)
    neg_beta_v = -beta * all_vals
    max_neg = neg_beta_v.max()
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_v - max_neg)))
    soft_min_val = -log_sum / beta
    return -soft_min_val


def _soft_min_grad(x, beta):
    """Gradient of the soft-min objective w.r.t. x (vectorized)."""
    x = x.reshape(-1, 2)
    n = x.shape[0]
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    bound_vals = 1.0 - radii
    all_vals = np.concatenate([pair_vals.ravel(), bound_vals])
    
    if all_vals.min() < 0:
        return np.zeros_like(x)
    
    neg_beta_v = -beta * all_vals
    max_neg = neg_beta_v.max()
    exp_terms = np.exp(neg_beta_v - max_neg)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(x)
    
    # Boundary part
    n_pairs = n * (n - 1)
    bound_weights = weights[n_pairs:]
    for i in range(n):
        r = radii[i]
        if r > 1e-10:
            grad[i] += bound_weights[i] * x[i] / r
    
    # Pairwise part (vectorized)
    pair_weights = weights[:n_pairs].reshape(n, n)
    # d v_ij / d x_i = (x_i - x_j) / (2 * d_ij)
    # grad[i] = sum_j w_ij * (x_i - x_j) / (2 * d_ij)
    # Use broadcasting: for each i, sum over j
    # (x_i - x_j) / (2 * d_ij) = diff[i,j] / (2 * dists[i,j])
    unit_vecs = diff / (2.0 * dists)  # shape (n, n, 2)
    unit_vecs[np.diag_indices(n)] = 0.0
    # grad[i] = sum_j pair_weights[i,j] * unit_vecs[i,j]
    grad += np.einsum('ij,ijk->k', pair_weights, unit_vecs)
    
    return -grad


def _true_score(x):
    """Compute the true score: min(min_pairwise/2, min_boundary)."""
    x = x.reshape(-1, 2)
    n = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    min_bound = (1.0 - radii).min()
    return min(min_pair, min_bound)


def _active_set_polish(x, max_iter=50, tol=1e-12):
    """
    Active-set Newton polish: identify the tightest constraints and solve
    the equality-constrained problem where those constraints are active.
    This gives a high-precision local optimum.
    """
    x = x.copy().reshape(-1, 2)
    n = x.shape[0]
    
    for it in range(max_iter):
        # Compute all constraint values
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        radii = np.sqrt(np.sum(x ** 2, axis=1))
        bound_vals = 1.0 - radii
        r = min(pair_vals.min(), bound_vals.min())
        
        # Find active constraints (within tolerance of r)
        tol_active = max(tol * 10, 1e-10)
        active_pairs = np.argwhere(pair_vals <= r + tol_active)
        active_bounds = np.where(bound_vals <= r + tol_active)[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Build the KKT system for maximizing r subject to active constraints
        # We solve: find dx that improves r while keeping active constraints satisfied
        # Use a simple projected gradient / Newton step
        
        # Gradient of r w.r.t. x: sum of gradients of active constraints
        grad_r = np.zeros_like(x)
        
        # Boundary contributions
        for i in active_bounds:
            ri = radii[i]
            if ri > 1e-10:
                grad_r[i] += x[i] / ri  # d(1-|x_i|)/dx_i = -x_i/|x_i|, we want to increase r
        
        # Pairwise contributions
        for (i, j) in active_pairs:
            if i >= j:
                continue  # avoid double counting
            d_ij = dists[i, j]
            if d_ij < 1e-10:
                continue
            # d(d_ij/2)/dx_i = (x_i - x_j)/(2*d_ij)
            vec = (x[i] - x[j]) / (2.0 * d_ij)
            grad_r[i] += vec
            grad_r[j] -= vec
        
        # Normalize
        gn = np.linalg.norm(grad_r)
        if gn < 1e-15:
            break
        
        # Line search: move in direction of grad_r
        alpha = tol * 10
        for trial_alpha in [alpha, alpha*0.5, alpha*0.25, alpha*0.1]:
            x_trial = x + trial_alpha * grad_r / gn
            # Check if all constraints still satisfied
            diff_t = x_trial[:, np.newaxis, :] - x_trial[np.newaxis, :, :]
            dists_t = np.sqrt(np.sum(diff_t ** 2, axis=2))
            np.fill_diagonal(dists_t, np.inf)
            pair_t = dists_t.min() / 2.0
            rad_t = np.sqrt(np.sum(x_trial ** 2, axis=1))
            bound_t = (1.0 - rad_t).min()
            r_trial = min(pair_t, bound_t)
            if r_trial > r - tol:
                x = x_trial
                break
        
        # Check convergence
        if abs(r_trial - r) < tol:
            break
    
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Try multiple initializations and pick the best
    inits = [_hex_initial(N), _ring_initial(N), _random_initial(N, seed=42), _random_initial(N, seed=123)]
    
    best_x = None
    best_r = 0.0
    for init in inits:
        r_init = _true_score(init)
        if np.isfinite(r_init) and r_init > best_r:
            best_r = r_init
            best_x = init.copy()
    
    x0_flat = best_x.ravel()
    
    # Annealing schedule: start with low beta (smooth), increase to high beta (sharp)
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1200.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                x0_flat,
                args=(beta,),
                method='L-BFGS-B',
                jac=_soft_min_grad,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            x_opt = res.x.reshape(-1, 2)
            r_opt = _true_score(x_opt)
            
            if np.isfinite(r_opt) and r_opt > best_r:
                best_r = r_opt
                best_x = x_opt.copy()
                x0_flat = x_opt.ravel()
        except Exception:
            pass
        
        if time.time() > deadline - 25.0:
            break
    
    # Final polish with a high beta if time permits
    if time.time() < deadline - 15.0:
        try:
            res = minimize(
                _soft_min_objective,
                x0_flat,
                args=(2000.0,),
                method='L-BFGS-B',
                jac=_soft_min_grad,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
            )
            x_opt = res.x.reshape(-1, 2)
            r_opt = _true_score(x_opt)
            if np.isfinite(r_opt) and r_opt > best_r:
                best_r = r_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Active-set Newton polish for higher precision
    if time.time() < deadline - 5.0:
        try:
            x_polished = _active_set_polish(best_x, max_iter=30, tol=1e-13)
            r_polished = _true_score(x_polished)
            if np.isfinite(r_polished) and r_polished > best_r:
                best_r = r_polished
                best_x = x_polished.copy()
        except Exception:
            pass
    
    # Final validation: ensure all points are inside and no coincident centres
    x_final = best_x
    radii = np.sqrt(np.sum(x_final ** 2, axis=1))
    for i in range(len(x_final)):
        if radii[i] > 1.0:
            x_final[i] = x_final[i] / radii[i] * 0.999
    
    return x_final
# EVOLVE-BLOCK-END
