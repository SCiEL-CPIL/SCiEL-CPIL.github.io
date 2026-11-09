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


def _energy_obj(x, p=6, eps=1e-8):
    """
    Inverse-power-sum energy: sum over all pairs of 1/(d_ij^p) + wall penalty.
    Minimizing this drives all distances up, with close pairs penalized heavily.
    """
    c = x.reshape(-1, 2)
    
    # Wall clearance penalty: for each point, distance to nearest wall
    # We want all points to be at least r from walls
    # Penalty: sum of 1/(wall_clearance^p) for each coordinate
    wall_x = HALF - np.abs(c[:, 0])
    wall_y = HALF - np.abs(c[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    wall_penalty = np.sum(1.0 / (wall_min + eps)**p)
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Inverse power sum over all pairs (upper triangle to avoid double counting)
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_energy = np.sum(1.0 / (pair_dist + eps)**p)
    
    return float(wall_penalty + pair_energy)


def _energy_grad(x, p=6, eps=1e-8):
    """
    Gradient of the inverse-power-sum energy.
    """
    c = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradient
    wall_x = HALF - np.abs(c[:, 0])
    wall_y = HALF - np.abs(c[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    
    # d/dx_i of 1/(wall_min_i + eps)^p = -p * sign * (wall_min_i + eps)^(-p-1)
    # where sign is +1 if wall_x < wall_y (limiting in x), -1 otherwise
    for i in range(N):
        if wall_x[i] < wall_y[i]:
            # Limited by x-wall
            w = wall_x[i] + eps
            dx = -p * np.sign(c[i, 0]) * w**(-p-1)
            grad[2*i] = dx
            grad[2*i+1] = 0.0
        else:
            # Limited by y-wall
            w = wall_y[i] + eps
            grad[2*i] = 0.0
            grad[2*i+1] = -p * np.sign(c[i, 1]) * w**(-p-1)
    
    # Pairwise gradient
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, 0.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-16))
    
    # For pair (i,j): d/dx_i of 1/(d_ij + eps)^p = -p * (d_ij + eps)^(-p-1) * (x_i - x_j)/d_ij
    # Sum over j != i
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dij = dist[i, j]
            if dij < 1e-12:
                continue
            factor = -p * (dij + eps)**(-p-1) / dij
            grad[2*i] += factor * d[i, j, 0]
            grad[2*i+1] += factor * d[i, j, 1]
    
    return grad


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


def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)
    best_energy = _energy_obj_fast(c0.flatten(), p=6)

    p_power = 6  # exponent for inverse power

    restarts = 0
    while time.time() - t_start < deadline:
        restarts += 1
        # Perturbation magnitude shrinks over restarts
        scale = 0.012 / (1.0 + 0.1 * restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)

        # Ensure no coincident centres
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()

        rem = deadline - (time.time() - t_start)
        if rem < 3.0:
            break

        x0 = c_start.flatten()
        bounds = [(-HALF + 1e-6, HALF - 1e-6)] * (2 * N)

        try:
            result = minimize(
                _energy_obj_fast,
                x0,
                jac=_energy_grad_fast,
                args=(p_power,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            
            if result.x is not None and np.all(np.isfinite(result.x)):
                c_opt = result.x.reshape(-1, 2)
                c_opt = np.clip(c_opt, -HALF + 1e-6, HALF - 1e-6)
                r_opt = _radius_of(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt
                    best_energy = _energy_obj_fast(best_c.flatten(), p=p_power)
        except Exception:
            pass

    # Final high-precision polish pass with tighter tolerances
    rem = deadline - (time.time() - t_start)
    if rem > 5.0:
        x0 = best_c.flatten()
        bounds = [(-HALF + 1e-8, HALF - 1e-8)] * (2 * N)
        try:
            result = minimize(
                _energy_obj_fast,
                x0,
                jac=_energy_grad_fast,
                args=(p_power,),
                method='L-BFGS-B',
                bounds=bounds,
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
