"""
Packing 200 equal circles in a unit square using inverse-power-sum energy minimization.
The key insight from the inspiration: switch from non-smooth soft-min to a smooth
inverse-power-sum objective that provides well-defined gradients everywhere.
This drives the system to maximize the minimum distance by penalizing close pairs heavily.
"""
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
    r0 = 0.0366
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
    # Fallback: uniform grid
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _energy_obj_fast(x, p=6, eps=1e-8):
    """Vectorized version of energy objective."""
    c = x.reshape(-1, 2)
    
    # Wall penalty
    wall_x = HALF - np.abs(c[:, 0])
    wall_y = HALF - np.abs(c[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    wall_penalty = np.sum(1.0 / (wall_min + eps)**p)
    
    # Pairwise
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_energy = np.sum(1.0 / (pair_dist + eps)**p)
    
    return float(wall_penalty + pair_energy)


def _energy_grad_fast(x, p=6, eps=1e-8):
    """Vectorized gradient."""
    c = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradient (vectorized)
    wall_x = HALF - np.abs(c[:, 0])
    wall_y = HALF - np.abs(c[:, 1])
    
    # Determine which wall is limiting
    x_limited = wall_x < wall_y
    
    # For x-limited points
    if np.any(x_limited):
        w = wall_x[x_limited] + eps
        grad[2*x_limited.nonzero()[0]] = -p * np.sign(c[x_limited, 0]) * w**(-p-1)
    
    # For y-limited points
    if np.any(~x_limited):
        y_limited = ~x_limited
        w = wall_y[y_limited] + eps
        grad[2*y_limited.nonzero()[0]+1] = -p * np.sign(c[y_limited, 1]) * w**(-p-1)
    
    # Pairwise gradient (vectorized)
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    # factor[i,j] = -p * (dist[i,j] + eps)^(-p-1) / dist[i,j]
    factor = -p * (dist + eps)**(-p-1) / dist
    np.fill_diagonal(factor, 0.0)
    
    # grad[2*i] = sum_j factor[i,j] * d[i,j,0]
    grad[0::2] = np.sum(factor * d[:, :, 0], axis=1)
    grad[1::2] = np.sum(factor * d[:, :, 1], axis=1)
    
    return grad


def _soft_min_obj(x, temperature=0.01):
    """
    Soft-min surrogate for the packing radius.
    Uses log-sum-exp to create a smooth approximation of min(d_ij, wall_i).
    """
    c = x.reshape(-1, 2)
    
    # Wall clearances
    wall_x = HALF - np.abs(c[:, 0])
    wall_y = HALF - np.abs(c[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    
    # Pairwise distances (half-distance since radius = d/2)
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu] / 2.0  # radius = distance/2
    
    # Combine all constraints: wall_min_i and pair_dist_ij
    all_constraints = np.concatenate([wall_min, pair_dist])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-x_i/T))) * T
    # This approximates min(x_i) as T -> 0
    shifted = -all_constraints / temperature
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -temperature * log_sum
    
    return -float(soft_min)  # We want to maximize, so minimize negative


def _soft_min_grad(x, temperature=0.01):
    """
    Gradient of the soft-min objective.
    """
    c = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall clearances
    wall_x = HALF - np.abs(c[:, 0])
    wall_y = HALF - np.abs(c[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    
    # Pairwise distances (half-distance)
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu] / 2.0
    
    all_constraints = np.concatenate([wall_min, pair_dist])
    
    # Soft-min gradient: d/dx_i of -T*log(sum(exp(-x_j/T)))
    # = sum_j (exp(-x_j/T) / sum_k(exp(-x_k/T))) * dx_j/dx_i
    shifted = -all_constraints / temperature
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Wall gradients
    x_limited = wall_x < wall_y
    wall_weights = weights[:N]
    
    # For x-limited points
    if np.any(x_limited):
        idx = x_limited.nonzero()[0]
        grad[2*idx] += -wall_weights[idx] * np.sign(c[idx, 0])
    
    # For y-limited points
    if np.any(~x_limited):
        idx = (~x_limited).nonzero()[0]
        grad[2*idx+1] += -wall_weights[idx] * np.sign(c[idx, 1])
    
    # Pairwise gradients
    pair_weights = weights[N:]
    # Reconstruct pair indices
    pair_i, pair_j = iu
    for k in range(len(pair_i)):
        w = pair_weights[k]
        i, j = pair_i[k], pair_j[k]
        dij = dist[i, j]
        if dij < 1e-12:
            continue
        # d(pair_dist)/d(c_i) = (c_i - c_j) / (2 * d_ij)
        # But we're minimizing -soft_min, so gradient sign is flipped
        grad[2*i] += w * (c[i, 0] - c[j, 0]) / dij
        grad[2*i+1] += w * (c[i, 1] - c[j, 1]) / dij
        grad[2*j] -= w * (c[i, 0] - c[j, 0]) / dij
        grad[2*j+1] -= w * (c[i, 1] - c[j, 1]) / dij
    
    return grad


def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    bounds = [(-HALF + 1e-6, HALF - 1e-6)] * (2 * N)

    # Phase 1: Use inverse-power-sum with moderate power for initial optimization
    p_power = 4
    restarts = 0
    max_restarts_phase1 = 8
    
    while time.time() - t_start < deadline and restarts < max_restarts_phase1:
        restarts += 1
        scale = 0.015 / (1.0 + 0.2 * restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)

        # Ensure no coincident centres
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()

        rem = deadline - (time.time() - t_start)
        if rem < 5.0:
            break

        x0 = c_start.flatten()

        try:
            result = minimize(
                _energy_obj_fast,
                x0,
                jac=_energy_grad_fast,
                args=(p_power,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            
            if result.x is not None and np.all(np.isfinite(result.x)):
                c_opt = result.x.reshape(-1, 2)
                c_opt = np.clip(c_opt, -HALF + 1e-6, HALF - 1e-6)
                r_opt = _radius_of(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt
        except Exception:
            pass

    # Phase 2: Use higher power for exploitation
    rem = deadline - (time.time() - t_start)
    if rem > 10.0:
        p_power = 8
        restarts = 0
        max_restarts_phase2 = 6
        
        while time.time() - t_start < deadline and restarts < max_restarts_phase2:
            restarts += 1
            scale = 0.008 / (1.0 + 0.15 * restarts)
            c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
            c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)

            d = c_start[:, None, :] - c_start[None, :, :]
            dist = np.sqrt(np.sum(d * d, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 1e-4:
                c_start = best_c.copy()

            rem = deadline - (time.time() - t_start)
            if rem < 5.0:
                break

            x0 = c_start.flatten()

            try:
                result = minimize(
                    _energy_obj_fast,
                    x0,
                    jac=_energy_grad_fast,
                    args=(p_power,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'maxfun': 12000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                
                if result.x is not None and np.all(np.isfinite(result.x)):
                    c_opt = result.x.reshape(-1, 2)
                    c_opt = np.clip(c_opt, -HALF + 1e-6, HALF - 1e-6)
                    r_opt = _radius_of(c_opt)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_c = c_opt
            except Exception:
                pass

    # Phase 3: Final polish with very tight tolerances
    rem = deadline - (time.time() - t_start)
    if rem > 8.0:
        x0 = best_c.flatten()
        bounds_tight = [(-HALF + 1e-8, HALF - 1e-8)] * (2 * N)
        try:
            result = minimize(
                _energy_obj_fast,
                x0,
                jac=_energy_grad_fast,
                args=(12,),  # Very high power for final polish
                method='L-BFGS-B',
                bounds=bounds_tight,
                options={'maxiter': 5000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                c_opt = result.x.reshape(-1, 2)
                c_opt = np.clip(c_opt, -HALF + 1e-8, HALF - 1e-8)
                r_opt = _radius_of(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt
        except Exception:
            pass

    # Final safety: ensure validity
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Verify no coincident centers
    d = best_c[:, None, :] - best_c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 1e-6:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    
    return best_c


if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
