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
    n = x.shape[0]
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


def _active_set_polish(x, max_iter=200, tol=1e-15):
    """
    High-precision active-set polish: at each iteration, find the k tightest
    constraints, build the exact KKT system (dual variables for active constraints),
    and solve the resulting linear system for the optimal step + radius increment.
    """
    x = x.copy().reshape(-1, 2)
    n = x.shape[0]
    dim = 2 * n
    best_r = _true_score(x)
    
    for it in range(max_iter):
        # Compute all constraint values
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        radii = np.sqrt(np.sum(x ** 2, axis=1))
        bound_vals = 1.0 - radii
        r = min(pair_vals.min(), bound_vals.min())
        
        if not np.isfinite(r):
            break
        
        # Find the k tightest constraints
        k = min(50, n * (n - 1) // 2 + n)
        tol_active = max(1e-12, 1e-10 * r)
        
        # Collect all constraint values with indices
        # Pair constraints: index = i * n + j for i < j
        # Bound constraints: index = n * (n - 1) // 2 + i
        n_pairs = n * (n - 1) // 2
        all_vals = np.empty(n_pairs + n)
        all_vals[:n_pairs] = pair_vals[np.triu_indices(n, k=1)].ravel()
        all_vals[n_pairs:] = bound_vals
        
        # Get indices of tightest constraints
        tight_idx = np.argsort(all_vals)[:k]
        tight_vals = all_vals[tight_idx]
        
        # Filter to only those within tolerance of r
        mask = tight_vals <= r + tol_active
        tight_idx = tight_idx[mask]
        tight_vals = tight_vals[mask]
        
        m = len(tight_idx)
        if m == 0:
            break
        
        # Build gradient matrix G (m x dim) where G[t, :] = grad of constraint t
        G = np.zeros((m, dim))
        
        for t, idx in enumerate(tight_idx):
            if idx < n_pairs:
                # Pair constraint
                # Convert linear index to (i, j)
                # For upper triangle: idx = i*n - i*(i+1)//2 + j - i - 1
                # Easier: iterate
                i = 0
                while True:
                    j_start = i + 1
                    j_end = n
                    count = j_end - j_start
                    if idx < count:
                        j = j_start + idx
                        break
                    idx -= count
                    i += 1
                    if i >= n - 1:
                        break
                if i >= n - 1:
                    continue
                d_ij = dists[i, j]
                if d_ij < 1e-12:
                    continue
                vec = (x[i] - x[j]) / d_ij
                G[t, 2 * i] += vec[0]
                G[t, 2 * i + 1] += vec[1]
                G[t, 2 * j] -= vec[0]
                G[t, 2 * j + 1] -= vec[1]
            else:
                # Bound constraint for circle idx - n_pairs
                i = idx - n_pairs
                ri = radii[i]
                if ri < 1e-12:
                    continue
                G[t, 2 * i] = x[i][0] / ri
                G[t, 2 * i + 1] = x[i][1] / ri
        
        # Check if G has full row rank
        try:
            # Solve KKT system:
            # [G  1] [lambda]   [0]
            # [1' 0] [dx]     = [1]
            # where 1 is a column vector of ones
            # G * lambda + dx = 0  => dx = -G * lambda
            # 1' * dx + dr = 1  => -1' * G * lambda + dr = 1
            # 
            # Actually the KKT for max dr s.t. G dx + g0 >= 0 (active constraints)
            # with dx being the step: we want to maximize dr such that
            # for each active constraint: G_t · dx >= -g_t (where g_t = val_t - r)
            # 
            # Simpler: solve the equality-constrained version where we set
            # G dx = 0 (keep active constraints exactly at their current value)
            # and maximize dr = min of constraint values after step.
            # 
            # The optimal direction is: find dx that maximizes the minimum
            # constraint value. For active constraints, this is:
            # maximize min_t (G_t · dx + g_t)
            # 
            # By LP duality, the optimal dx is in the null space of G
            # and the optimal dr is found by solving:
            # G dx = 0, maximize c' dx where c is the dual vector.
            # 
            # Practical approach: solve G dx = 0 (minimum norm) and check.
            # But that gives dx=0. Instead, we want the direction that
            # increases the minimum constraint.
            # 
            # Better: solve the QP: min ||dx||^2 / 2 s.t. G dx >= -g
            # where g = tight_vals - r (all >= 0, some = 0)
            # 
            # For the active set where g_t = 0: G_t dx >= 0
            # For near-active where g_t > 0: G_t dx >= -g_t
            # 
            # The KKT system for this QP:
            # I dx + G' lambda = 0
            # G dx + g - mu = 0  (complementary slackness: lambda * (G dx + g - mu) = 0)
            # 
            # This is complex. Let's use a simpler approach:
            # Project the "ideal" direction (gradient of r) onto the null space
            # of the most active constraints.
            
            # Compute the gradient of r (sum of active constraint gradients)
            grad_r = G.sum(axis=0)  # sum of all active constraint gradients
            gn = np.linalg.norm(grad_r)
            if gn < 1e-15:
                break
            
            # Project grad_r onto null space of G (for the most active constraints)
            # Use SVD to find null space
            try:
                U, s, Vt = np.linalg.svd(G, full_matrices=True)
                # Null space is Vt[rank:]
                rank = np.sum(s > 1e-10 * s[0]) if len(s) > 0 else 0
                null_space = Vt[rank:]  # shape (dim - rank, dim)
                
                if null_space.shape[0] == 0:
                    # G has full column rank, no null space
                    # Use pseudo-inverse to find minimum norm solution
                    dx = np.linalg.lstsq(G, -G @ (grad_r / gn), rcond=None)[0]
                else:
                    # Project grad_r onto null space
                    proj = null_space.T @ null_space @ (grad_r / gn)
                    dx = proj
                    # Also add a small component in the range space to increase r
                    # Actually, just use the null space projection
                    if np.linalg.norm(dx) < 1e-15:
                        dx = np.linalg.lstsq(G, np.zeros(m), rcond=None)[0]
                        break
            except np.linalg.LinAlgError:
                dx = grad_r / gn
            
            # Normalize dx
            dxn = np.linalg.norm(dx)
            if dxn < 1e-15:
                break
            dx = dx / dxn
            
            # Line search: find best alpha
            best_alpha = 0.0
            best_r_trial = r
            for alpha in [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7]:
                x_trial = x + alpha * dx
                r_trial = _true_score(x_trial)
                if np.isfinite(r_trial) and r_trial > best_r_trial:
                    best_r_trial = r_trial
                    best_alpha = alpha
            
            if best_alpha == 0.0:
                break
            
            x = x + best_alpha * dx
            r = best_r_trial
            
            if r > best_r:
                best_r = r
            
        except Exception:
            break
        
        # Convergence check
        if r - best_r < tol * max(1, abs(best_r)):
            break
    
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Try multiple initializations and pick the best
    inits = [_hex_initial(N), _ring_initial(N), _random_initial(N, seed=42), _random_initial(N, seed=123), _random_initial(N, seed=999)]
    
    best_x = None
    best_r = 0.0
    for init in inits:
        r_init = _true_score(init)
        if np.isfinite(r_init) and r_init > best_r:
            best_r = r_init
            best_x = init.copy()
    
    x0_flat = best_x.ravel()
    
    # Annealing schedule: start with low beta (smooth), increase to high beta (sharp)
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1200.0, 2500.0, 5000.0, 10000.0, 20000.0]
    
    for beta in betas:
        if time.time() > deadline - 25.0:
            break
        
        try:
            # Use tighter tolerances for higher betas
            if beta < 100:
                opts = {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 4000}
            elif beta < 500:
                opts = {'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 3000}
            elif beta < 5000:
                opts = {'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            else:
                opts = {'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-11, 'maxfun': 1500}
            
            res = minimize(
                _soft_min_objective,
                x0_flat,
                args=(beta,),
                method='L-BFGS-B',
                jac=_soft_min_grad,
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
    
    # Active-set polish: run multiple times with different initializations
    if time.time() < deadline - 5.0:
        try:
            x_polished = _active_set_polish(best_x, max_iter=200, tol=1e-15)
            r_polished = _true_score(x_polished)
            if np.isfinite(r_polished) and r_polished > best_r:
                best_r = r_polished
                best_x = x_polished.copy()
        except Exception:
            pass
    
    # Second polish pass if time permits
    if time.time() < deadline - 2.0:
        try:
            x_polished2 = _active_set_polish(best_x, max_iter=100, tol=1e-16)
            r_polished2 = _true_score(x_polished2)
            if np.isfinite(r_polished2) and r_polished2 > best_r:
                best_r = r_polished2
                best_x = x_polished2.copy()
        except Exception:
            pass
    
    # Final validation: ensure all points are inside and no coincident centres
    x_final = best_x.copy()
    radii = np.sqrt(np.sum(x_final ** 2, axis=1))
    for i in range(len(x_final)):
        if radii[i] > 1.0:
            x_final[i] = x_final[i] / radii[i] * 0.999
    
    return x_final
# EVOLVE-BLOCK-END
