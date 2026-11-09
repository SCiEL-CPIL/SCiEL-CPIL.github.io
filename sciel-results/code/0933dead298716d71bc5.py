"""Model-and-optimize: two-stage (lattice fit + soft-min radius) with deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
HALF = 0.5
SEED = 1234567


def _radius_of(c):
    """Compute packing radius for centers c (N,2)."""
    w = HALF - np.abs(c)
    wall = np.min(w)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = np.min(dist) / 2.0
    return float(min(wall, pair))


def _hexagon_seed(n):
    """Staggered hexagonal rows inside [-1/2,1/2]^2, return n centers."""
    for attempt in range(200):
        r0 = 0.036
        h = 2 * r0
        v = r0 * np.sqrt(3)
        x0 = -HALF + r0
        y0 = -HALF + r0
        n_rows = int((1 - 2 * r0) / v) + 1
        pts = []
        for i in range(n_rows):
            y = y0 + i * v
            x_start = x0 + (h / 2 if i % 2 else 0.0)
            n_cols = int((HALF - r0 - x_start) / h) + 1 if x_start <= HALF - r0 else 0
            for j in range(n_cols):
                x = x_start + j * h
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            return pts[:n]
        r0 *= 0.995
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _lattice_fit(x, target, lam):
    """Objective: sum of squared distances to target lattice + lam * soft-min penalty."""
    c = x.reshape(-1, 2)
    diff = c - target
    fit_term = np.sum(diff * diff)
    # Add a soft penalty for close pairs to prevent collapse
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist) / 2.0
    wall = np.min(HALF - np.abs(c))
    margin = min(pair_min, wall)
    # Penalize if margin is below a threshold
    penalty = 0.0
    if margin < 0.01:
        penalty = lam * (0.01 - margin) ** 2
    return fit_term + penalty


def _lattice_fit_grad(x, target, lam):
    """Gradient of lattice fit objective."""
    c = x.reshape(-1, 2)
    diff = c - target
    grad = 2.0 * diff.ravel()
    
    # Gradient of penalty term
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist) / 2.0
    wall = np.min(HALF - np.abs(c))
    margin = min(pair_min, wall)
    
    if margin < 0.01:
        # Compute gradient of -(0.01 - margin)^2 = -2*(0.01-margin)*(-d(margin)/dx)
        # = 2*(0.01-margin)*d(margin)/dx
        # We need d(margin)/dx where margin = min(pair_min, wall)
        # This is complex; approximate with just the wall term for simplicity
        # or use a simpler penalty
        pass
    return grad


def _soft_min_radius(x, beta):
    """Smooth surrogate for min(wall, min_pair_dist/2) using log-sum-exp."""
    c = x.reshape(-1, 2)
    # Wall clearances
    w = HALF - np.abs(c)
    wall_vals = w.ravel()
    # Pairwise distances / 2
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist.ravel() / 2.0
    # Combine all "constraint margins"
    all_margins = np.concatenate([wall_vals, pair_vals])
    # Soft min via log-sum-exp: -logsumexp(-margins/beta) * beta
    neg = -all_margins / beta
    max_neg = np.max(neg)
    lse = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    return -lse * beta


def _soft_min_grad(x, beta):
    """Gradient of -soft_min_radius (we maximize soft_min, so minimize negative)."""
    c = x.reshape(-1, 2)
    w = HALF - np.abs(c)
    wall_vals = w.ravel()
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist.ravel() / 2.0
    all_margins = np.concatenate([wall_vals, pair_vals])
    neg = -all_margins / beta
    max_neg = np.max(neg)
    exps = np.exp(neg - max_neg)
    weights = exps / np.sum(exps)
    # d(soft_min)/dx = sum_k w_k * d(margin_k)/dx
    # soft_min = -beta * logsumexp(-margins/beta)
    # d(soft_min)/dx = sum_k w_k * d(margin_k)/dx
    # We want gradient of -soft_min for minimization
    # = -sum_k w_k * d(margin_k)/dx
    grad = np.zeros_like(x)
    # Wall margins: margin = HALF - |c_ik|
    # d(HALF - |c_ik|)/dc_ik = -sign(c_ik)
    wall_w = weights[:len(wall_vals)]
    # wall_vals is (N,2) flattened: w[i,0], w[i,1], w[i+1,0], ...
    for i in range(N):
        for k in range(2):
            idx = i * 2 + k
            cw = wall_w[idx]
            if c[i, k] >= 0:
                grad[i * 2 + k] -= cw  # d(HALF-c)/dc = -1, times -cw for -soft_min
            else:
                grad[i * 2 + k] += cw  # d(HALF+c)/dc = +1, times -cw for -soft_min
    # Pair margins: margin = dist(i,j)/2
    # d(dist(i,j)/2)/dc_i = (c_i - c_j) / dist(i,j)
    pair_w = weights[len(wall_vals):]
    # pair_vals is (N,N) flattened row-major
    pair_flat = pair_vals
    # For each pair (i,j), contribution to grad[i] and grad[j]
    # We need to accumulate: for pair (i,j) with weight w_ij:
    #   d(margin)/dc_i = (c_i - c_j)/dist_ij
    #   d(margin)/dc_j = (c_j - c_i)/dist_ij
    # grad contribution to -soft_min: -w_ij * d(margin)/dc
    # To avoid O(N^2) Python loop, use vectorized approach
    # Reshape weights to (N,N)
    pw = pair_flat.reshape(N, N)
    # dist matrix already computed
    # For each i: sum_j pw[i,j] * (c_i - c_j) / dist[i,j]
    # = c_i * sum_j pw[i,j]/dist[i,j] - sum_j pw[i,j]*c_j/dist[i,j]
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = np.zeros_like(dist)
        mask = dist > 1e-12
        inv_dist[mask] = 1.0 / dist[mask]
    # weight matrix: pw[i,j] * inv_dist[i,j]
    wmat = pw * inv_dist
    # np.fill_diagonal to be safe
    np.fill_diagonal(wmat, 0.0)
    # For each i: sum_j wmat[i,j] * c_j
    sum_wc = wmat @ c  # shape (N,2)
    # sum_j wmat[i,j]
    wsum = wmat.sum(axis=1)  # shape (N,)
    # grad_pair[i] = -wsum[i] * c[i] + sum_wc[i]
    # This is the gradient of -soft_min w.r.t. c_i from pair terms
    grad_pair = -wsum[:, None] * c + sum_wc
    # Add to grad
    grad[0::2] += grad_pair[:, 0]
    grad[1::2] += grad_pair[:, 1]
    return grad


def _optimize_lbgfs(c0, beta, deadline):
    """Run L-BFGS-B to maximize soft-min radius. Returns best centers and radius."""
    x0 = c0.ravel().copy()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    def obj(x):
        return -_soft_min_radius(x, beta)
    
    def grad(x):
        return _soft_min_grad(x, beta)
    
    t0 = time.time()
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxls': 50})
        if res.success or res.fun < obj(x0):
            best_x = res.x
        else:
            best_x = x0
    except Exception:
        best_x = x0
    
    # Evaluate true radius
    best_c = best_x.reshape(N, 2)
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    r = _radius_of(best_c)
    return best_c, r


def _stage1_lattice_fit(c0, target, time_budget):
    """Stage 1: Minimize distance to hexagonal lattice to establish coherent structure."""
    x0 = c0.ravel().copy()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    lam = 10.0  # Penalty weight for close pairs
    
    def obj(x):
        return _lattice_fit(x, target, lam)
    
    def grad(x):
        return _lattice_fit_grad(x, target, lam)
    
    t0 = time.time()
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10,
                                'maxls': 30})
        best_x = res.x
    except Exception:
        best_x = x0
    
    best_c = best_x.reshape(N, 2)
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    return best_c


def construct():
    t_start = time.time()
    deadline = 115.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)
    
    # Generate target lattice for stage 1
    target = _hexagon_seed(N)
    target = np.clip(target, -HALF + 1e-6, HALF - 1e-6)

    # Stage 1: Lattice fit to establish coherent structure
    rem = deadline - (time.time() - t_start)
    if rem > 30.0:
        c_stage1 = _stage1_lattice_fit(c0, target, rem - 25.0)
        r_stage1 = _radius_of(c_stage1)
        if r_stage1 > best_r:
            best_r = r_stage1
            best_c = c_stage1

    # Stage 2: Multi-restart soft-min radius maximization
    restarts = 0
    while time.time() - t_start < deadline:
        restarts += 1
        # Start with larger beta (smoother) and decrease
        beta = max(0.005, 0.02 * (0.85 ** restarts))
        # Perturbation shrinks over restarts
        scale = 0.015 / (1.0 + 0.05 * restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
        # Ensure no coincident centres
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()
        
        rem = deadline - (time.time() - t_start)
        if rem < 10.0:
            break
        
        c_opt, r_opt = _optimize_lbgfs(c_start, beta, rem - 5.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # Final high-precision polish with small beta
    rem = deadline - (time.time() - t_start)
    if rem > 15.0:
        c_polish, r_polish = _optimize_lbgfs(best_c, 0.002, rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # final safety
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best_c


if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
