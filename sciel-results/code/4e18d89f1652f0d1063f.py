"""Improved 50 equal circles in a unit square using soft-min + SA + gradient polish."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 142.0

def _compute_radius(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    if min_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_pair, min_wall)

def _soft_min_obj(x, tau):
    centers = x.reshape(-1, 2)
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    all_dists = np.concatenate([walls, pair_dists])
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    return -(min_d - tau * log_sum)

def _soft_min_grad(x, tau):
    centers = x.reshape(-1, 2)
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    all_dists = np.concatenate([walls, pair_upper])
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    grad = np.zeros_like(x)
    for i in range(N):
        w = weights[i]
        if walls[i] == wall_x[i]:
            grad[2*i] += w * np.sign(centers[i, 0])
        if walls[i] == wall_y[i]:
            grad[2*i+1] += w * np.sign(centers[i, 1])
    for k in range(len(iu)):
        w = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[i, j]
        if d > 1e-12:
            factor = 0.5 * w / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    return grad

def _initial_grid():
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for y in ticks:
        for x in ticks:
            pts.append([x, y])
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    grid = np.array(pts[:N])
    rng = np.random.default_rng(42)
    grid += rng.normal(0, 0.002, grid.shape)
    return np.clip(grid, -0.49, 0.49)

def _force_relax(centers, time_budget):
    start = time.time()
    cur = centers.copy()
    best = cur.copy()
    best_r = _compute_radius(cur)
    momentum = np.zeros_like(cur)
    step = 0.015
    damping = 0.85
    for _ in range(8000):
        if time.time() - start > time_budget:
            break
        diff = cur[:, np.newaxis, :] - cur[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e6)
        inv_d2 = 0.4 / (dists**2 + 1e-8)
        unit = diff / (dists[:, :, np.newaxis] + 1e-8)
        forces = np.sum(inv_d2[:, :, np.newaxis] * unit, axis=1)
        wf = np.zeros_like(cur)
        wl = cur[:, 0] + 0.5
        wr = 0.5 - cur[:, 0]
        wf[:, 0] += 1.0 / (wl**2 + 1e-6) * (wl < 0.08)
        wf[:, 0] -= 1.0 / (wr**2 + 1e-6) * (wr < 0.08)
        wb = cur[:, 1] + 0.5
        wt = 0.5 - cur[:, 1]
        wf[:, 1] += 1.0 / (wb**2 + 1e-6) * (wb < 0.08)
        wf[:, 1] -= 1.0 / (wt**2 + 1e-6) * (wt < 0.08)
        momentum = damping * momentum + (forces + wf)
        cur += step * momentum
        cur = np.clip(cur, -0.5, 0.5)
        step *= 0.999
        r = _compute_radius(cur)
        if r > best_r:
            best_r = r
            best = cur.copy()
    return best, best_r

def _lbfgs(x0, tau, budget):
    start = time.time()
    def og(x):
        if time.time() - start > budget:
            return 0.0, np.zeros_like(x)
        return _soft_min_obj(x, tau), _soft_min_grad(x, tau)
    try:
        res = opt.minimize(og, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2*N),
                           options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
        return np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
    except:
        return x0.reshape(-1, 2)

def _sa_search(centers, time_budget, rng):
    start = time.time()
    cur = centers.copy()
    best = cur.copy()
    best_r = _compute_radius(cur)
    temp = 0.003
    cool = 0.995
    for _ in range(30000):
        if time.time() - start > time_budget:
            break
        i = rng.integers(N)
        trial = cur.copy()
        trial[i] += rng.normal(0, temp, 2)
        trial[i] = np.clip(trial[i], -0.5, 0.5)
        r_t = _compute_radius(trial)
        r_c = _compute_radius(cur)
        if r_t >= r_c or rng.random() < np.exp((r_t - r_c) / max(temp, 1e-6)):
            cur = trial
        if r_t > best_r:
            best_r = r_t
            best = cur.copy()
        temp *= cool
        if temp < 1e-6:
            temp = 1e-6
    return best, best_r

def _final_polish(centers, budget):
    start = time.time()
    tau = 0.0005
    x0 = centers.flatten()
    def og(x):
        if time.time() - start > budget:
            return 0.0, np.zeros_like(x)
        return _soft_min_obj(x, tau), _soft_min_grad(x, tau)
    try:
        res = opt.minimize(og, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2*N),
                           options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False})
        polished = np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
        if _compute_radius(polished) > _compute_radius(centers):
            return polished
    except:
        pass
    return centers

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = np.random.default_rng(999)
    best = _initial_grid()
    best_r = _compute_radius(best)
    
    # Force relax
    if time.time() < deadline - 70:
        best, best_r = _force_relax(best, min(40, deadline - time.time() - 70))
    
    # Multi-tau L-BFGS
    if time.time() < deadline - 50:
        x0 = best.flatten()
        for tau in [0.012, 0.008, 0.005, 0.003, 0.002]:
            if time.time() > deadline - 30:
                break
            trial = _lbfgs(x0, tau, min(10, deadline - time.time() - 25))
            tr = _compute_radius(trial)
            if tr > best_r:
                best_r = tr
                best = trial
            x0 = best.flatten()
    
    # SA from polished
    if time.time() < deadline - 25:
        best, best_r = _sa_search(best, min(18, deadline - time.time() - 20), rng)
    
    # Another L-BFGS pass
    if time.time() < deadline - 15:
        for tau in [0.004, 0.002, 0.001]:
            if time.time() > deadline - 10:
                break
            trial = _lbfgs(best.flatten(), tau, min(8, deadline - time.time() - 8))
            tr = _compute_radius(trial)
            if tr > best_r:
                best_r = tr
                best = trial
    
    # Final polish
    if time.time() < deadline - 5:
        best = _final_polish(best, min(8, deadline - time.time() - 3))
        best_r = _compute_radius(best)
    
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        return _initial_grid()
    return best
