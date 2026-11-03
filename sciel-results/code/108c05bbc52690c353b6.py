"""Pack 50 equal circles in a unit square, maximize common radius.

Combines hard-disk radius expansion (inspiration 1) with a soft-min L-BFGS-B
polish (the measured improvement strategy) and a QCQP-style active-set
refinement on the kissing graph.
"""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

N = 50
DEADLINE = 115.0  # safety margin before 150s kill

def _radius(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    # Wall constraint: min_i min_k (0.5 - |c_ik|)
    wall_r = np.min(0.5 - np.abs(c))
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_d = np.sqrt(d2.min())
    return min(wall_r, min_pair_d / 2.0)

def _radius_fast(c):
    """Slightly faster radius using squared distances where possible."""
    wall_r = np.min(0.5 - np.abs(c))
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_d2 = d2.min()
    pair_r = np.sqrt(min_pair_d2) / 2.0
    return min(wall_r, pair_r)

def _soft_min_radius(c, t=50.0):
    """
    Differentiable surrogate: soft-min of all pairwise half-distances and
    wall distances.  As t -> inf this approaches the true min radius.
    Used inside L-BFGS-B.
    """
    # Wall terms
    wall = 0.5 - np.abs(c)  # shape (N, 2)
    wall_flat = wall.ravel()
    # Pairwise terms
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    pair_terms = np.sqrt(d2) / 2.0
    pair_flat = pair_terms.ravel()
    
    all_terms = np.concatenate([wall_flat, pair_flat])
    # Soft-min: -t * log(sum(exp(-x/t)))
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    soft = m - t * np.log(s.sum())
    return soft

def _soft_min_radius_neg(x, t=50.0):
    """Negative soft-min radius for minimization."""
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t)

def _soft_min_radius_grad(x, t=50.0):
    """Gradient of negative soft-min radius."""
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    pair_terms = np.sqrt(d2) / 2.0
    pair_flat = pair_terms.ravel()
    
    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()
    
    # d(-soft)/dx = (1/Z) * sum_j exp(-(x_j - m)/t) * dx_j/dx
    # We need gradients of each term w.r.t. c
    grad = np.zeros_like(c)
    
    # Wall gradients
    # term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    # wall_flat maps: index i*2+k -> c[i,k]
    w_grad = -sign_c.ravel()  # shape (2N,)
    grad_flat = np.zeros(2 * N)
    grad_flat[:2*N] += (exps[:2*N] / Z) * w_grad
    
    # Pair gradients
    # term_ij = ||c_i - c_j|| / 2
    # d/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, 1.0)  # avoid div by zero
    # For each pair (i,j), i<j:
    for i in range(N):
        for j in range(i+1, N):
            idx = i * N + j  # index in pair_flat for (i,j)
            # pair_flat is from pair_terms.ravel(), which is (N,N) row-major
            # pair_terms[i,j] and pair_terms[j,i] both exist
            idx_ij = i * N + j
            idx_ji = j * N + i
            w_ij = exps[2*N + idx_ij] / Z
            w_ji = exps[2*N + idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 0:
                dij = c[i] - c[j]
                norm = dist[i, j]
                if norm > 1e-12:
                    g = dij / (2.0 * norm)
                    grad[i] += total_w * g
                    grad[j] -= total_w * g
    
    return grad_flat.reshape(N, 2)

def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square.  Vectorized.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        # Overlap force: magnitude proportional to (2R - d) for d < 2R
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _active_set_refine(c, r, max_iter=200, tol=1e-10):
    """
    QCQP-style active-set refinement: identify the 'kissing' graph
    (pairs at distance ~2r and circles touching walls), then solve a
    local quadratic program to nudge centres so that all active
    constraints are satisfied with equality and r can be increased.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur
    
    for _ in range(max_iter):
        # Identify active constraints
        diff = c[:, None, :] - c[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        # Active pairs: distance close to 2*r_cur
        tol_d = 1e-4
        active_pairs = np.argwhere(d < 2 * r_cur + tol_d)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        # Active walls: circles close to a wall
        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i, k] < r_cur + tol_d:
                    active_walls.append((i, k))
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Build a small linear system: for each active pair (i,j),
        # we want to move i and j apart along the line connecting them
        # by a tiny amount to increase the minimum distance.
        # For each active wall (i,k), we want to move i inward.
        # This is a simplified active-set step.
        
        # Simpler: just do a small gradient step on the soft-min with
        # a very small temperature to focus on the active constraints.
        t_soft = max(1.0, 10 * tol_d / r_cur)
        x = c.ravel()
        res = minimize(
            _soft_min_radius_neg,
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t_soft),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 50, 'gtol': 1e-8}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > r_cur + 1e-12:
            c = c_new
            r_cur = r_new
        else:
            break
    
    return c, r_cur

def _init_7x7_plus():
    """7x7 grid (49 points) + 50th at centre, then jitter to remove duplicate."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    # (0,0) is at index (3,3) in the 7x7 grid
    # Add 50th point with a small offset
    extra = np.array([[0.02, 0.02]])
    return np.vstack([g, extra])

def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.02, 0.02, c.shape)
    return c

def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))

def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    """
    Gradually increase the target radius R from R0 to R1,
    running hard-disk relaxation at each step.
    """
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 10:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
        if (time.time() - t0) > DEADLINE - 15:
            break
    return best_c, best_r

def _lbfgs_polish(c, t0, time_budget=10.0):
    """
    L-BFGS-B polish using soft-min radius with decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)
    
    # Try a few temperature values, decreasing
    temps = [100.0, 50.0, 20.0, 10.0, 5.0]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-12}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > best_r:
            best_r = r_new
            best_x = res.x.copy()
            x = res.x.copy()
    
    return best_x.reshape(N, 2), best_r

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Multiple initializations
    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_random(rng),
    ]
    
    best_c = None
    best_r = -1.0
    
    for start in inits:
        if (time.time() - t0) > DEADLINE - 20:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 800, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)
    
    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 15:
        c_polished, r_polished = _lbfgs_polish(best_c, t0, time_budget=min(20.0, DEADLINE - 15 - (time.time() - t0)))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished
    
    # Phase 3: Active-set refinement on kissing graph
    if (time.time() - t0) < DEADLINE - 5:
        c_as, r_as = _active_set_refine(best_c, best_r, max_iter=50)
        if r_as > best_r:
            best_r = r_as
            best_c = c_as
    
    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_c
