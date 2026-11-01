"""Packing 300 equal circles in the unit disk via soft-min surrogate + L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
import time
import sys

def _hex_initial(n):
    """Generate n points in a hexagonal lattice inside the unit disk,
    choosing spacing so we get at least n points, then pick the n closest to center."""
    # Estimate spacing: area per point ~ pi / n, hex packing density ~ 0.9069
    # spacing ~ sqrt(4 * pi / (n * sqrt(3) * 0.9069))
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    # Try a few spacings and pick the one that gives >= n points with best min distance
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
            # Compute min pairwise distance
            from scipy.spatial.distance import pdist
            d = pdist(pts)
            md = d.min()
            if md > best_min_dist:
                best_min_dist = md
                best_points = pts
        spacing *= 0.97
    if best_points is None:
        # Fallback: concentric rings
        pts = np.zeros((n, 2))
        pts[0] = [0, 0]
        for i in range(1, n):
            ring = int(np.ceil(np.sqrt(i)))
            angle = 2 * np.pi * i / max(1, ring)
            r = 0.95 * np.sqrt(i / n)
            pts[i] = [r * np.cos(angle), r * np.sin(angle)]
        return pts
    return best_points


def _soft_min_objective(x, beta):
    """
    Objective: maximize r = min(min_pairwise_dist/2, min_boundary_margin)
    We minimize -r. Using soft-min:
      soft_min(a, b, beta) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    For pairwise: soft_min over all pair distances / 2
    For boundary: soft_min over (1 - |c_i|)
    Combine: soft_min(pairwise_soft, boundary_soft, beta2)
    
    Actually simpler: define f(x) = min over all constraints of the "radius" each constraint allows.
    Constraints: for pair (i,j): r <= d_ij / 2
                for point i: r <= 1 - |c_i|
    We want max r such that all satisfied. Equivalent to minimize -min(all constraints).
    
    Soft-min surrogate: S = -1/beta * log(sum(exp(-beta * c_k))) where c_k are all constraint values.
    Minimize S (which approximates -min(c_k) as beta->inf).
    """
    x = x.reshape(-1, 2)
    # Pairwise distances
    # For efficiency, compute all pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Pairwise constraint values: d_ij / 2
    pair_vals = dists / 2.0
    # Boundary constraint values: 1 - |c_i|
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    bound_vals = 1.0 - radii
    
    # Combine all constraint values
    all_vals = np.concatenate([pair_vals.ravel(), bound_vals])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * v)))
    # For numerical stability, subtract max
    v_min = all_vals.min()
    if v_min < 0:
        # All constraints violated; return a large penalty
        return -1000.0 + beta * abs(v_min)
    
    # Use log-sum-exp trick
    neg_beta_v = -beta * all_vals
    max_neg = neg_beta_v.max()
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_v - max_neg)))
    soft_min_val = -log_sum / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val


def _soft_min_grad(x, beta):
    """Gradient of the soft-min objective w.r.t. x."""
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
    
    # d(soft_min)/d(v_k) = exp(-beta*v_k) / sum(exp(-beta*v_j)) = exp_terms[k] / sum_exp
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(x)
    
    # Boundary part: v_k = 1 - |c_k| for k in [n_pairs, n_pairs+n)
    # d v_k / d x_k = -x_k / |x_k|
    n_pairs = n * (n - 1)
    bound_weights = weights[n_pairs:]
    # grad contribution from boundary: -sum_k bound_weights[k] * d v_k/d x_k = -sum_k w_k * (-x_k/r_k) = sum_k w_k * x_k/r_k
    for i in range(n):
        r = radii[i]
        if r > 1e-10:
            grad[i] += bound_weights[i] * x[i] / r
        # else: derivative of 1-|x| at 0 is undefined; use 0
    
    # Pairwise part: v_{ij} = d_ij / 2
    # d v_{ij} / d x_i = (x_i - x_j) / (2 * d_ij)
    # d v_{ij} / d x_j = (x_j - x_i) / (2 * d_ij)
    pair_weights = weights[:n_pairs].reshape(n, n)
    # For each pair (i,j) with i!=j:
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights[i, j]
            if dists[i, j] < 1e-10:
                continue
            # d v_ij / d x_i = (x_i - x_j) / (2 * d_ij)
            grad[i] += w * (x[i] - x[j]) / (2.0 * dists[i, j])
            # d v_ij / d x_j = (x_j - x_i) / (2 * d_ij)
            grad[j] += w * (x[j] - x[i]) / (2.0 * dists[i, j])
    
    # Objective is -soft_min, so gradient is -d(soft_min)/dx
    # d(soft_min)/dx = sum_k weights[k] * d(v_k)/dx
    # So d(objective)/dx = -sum_k weights[k] * d(v_k)/dx = -grad (as computed above)
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


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Initial point: hexagonal lattice
    x0 = _hex_initial(N)
    x0_flat = x0.ravel()
    
    # Verify initial is valid
    r0 = _true_score(x0)
    if not np.isfinite(r0) or r0 <= 0:
        # Fallback
        x0 = np.zeros((N, 2))
        for i in range(1, N):
            angle = 2 * np.pi * i / N
            r = 0.9 * np.sqrt(i / N)
            x0[i] = [r * np.cos(angle), r * np.sin(angle)]
        x0_flat = x0.ravel()
        r0 = _true_score(x0)
    
    best_x = x0.copy()
    best_r = r0
    
    # Annealing schedule: start with low beta (smooth), increase to high beta (sharp)
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1200.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        # Optimize with current beta
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
        except Exception as e:
            pass
        
        # If time is running low, do one more quick optimization with higher beta
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
    
    # Final validation: ensure all points are inside and no coincident centres
    x_final = best_x
    radii = np.sqrt(np.sum(x_final ** 2, axis=1))
    # Clip any points slightly outside
    for i in range(len(x_final)):
        if radii[i] > 1.0:
            x_final[i] = x_final[i] / radii[i] * 0.999
    
    return x_final
# EVOLVE-BLOCK-END
