"""
Inverse-power-sum energy minimization with L-BFGS-B for 200 equal circles in a unit square.

Key insight: instead of trying to directly maximize the min-distance (nonsmooth), we
minimize a smooth energy E = sum_{i<j} (1/d_ij^p) + sum_i sum_k (1/wall_ik^p) with large
p. This naturally pushes all distances up while heavily penalizing the smallest ones.
L-BFGS-B handles the box constraints efficiently.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
HALF = 0.5
SEED = 42
T = 150.0  # total wall-clock limit
MARGIN = 10.0  # safety margin


def _radius_of(c):
    """Compute the true packing radius for centers c (N, 2)."""
    w = HALF - np.abs(c)
    wall = np.min(w)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = np.min(dist) / 2.0
    return float(min(wall, pair))


def _hexagon_seed(n, r0=0.0366):
    """Staggered hexagonal rows inside [-1/2,1/2]^2."""
    pts = []
    h = 2 * r0
    v = r0 * np.sqrt(3)
    x0 = -HALF + r0
    y0 = -HALF + r0
    n_rows = int((1 - 2 * r0) / v) + 1
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
    # Fallback: grid
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _energy_and_grad(flat, p, eps=1e-12):
    """
    Compute inverse-power-sum energy and gradient.
    E = sum_{i<j} (1/d_ij^p) + sum_i sum_k (1/wall_ik^p)
    flat: (2N,) array of centers flattened.
    Returns (energy, grad_flat).
    """
    n = N
    c = flat.reshape(n, 2)
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, eps * eps)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Energy from pairs: sum_{i<j} (1/d_ij^p)
    # Using the full matrix, each pair counted twice, so divide by 2
    inv_dist_p = np.power(dist, -p)
    np.fill_diagonal(inv_dist_p, 0.0)
    pair_energy = 0.5 * np.sum(inv_dist_p)
    
    # Gradient from pairs:
    # d/d c_i [ (1/d_ij^p) ] = -p * (1/d_ij^(p+2)) * (c_i - c_j)
    # Sum over j != i
    # grad_pair_i = -p * sum_{j != i} (1/d_ij^(p+2)) * (c_i - c_j)
    inv_dist_p2 = np.power(dist, -(p + 2))
    np.fill_diagonal(inv_dist_p2, 0.0)
    # grad_pair[i] = -p * sum_j inv_dist_p2[i,j] * (c[i] - c[j])
    # = -p * ( c[i] * sum_j inv_dist_p2[i,j] - sum_j inv_dist_p2[i,j] * c[j] )
    row_sum = np.sum(inv_dist_p2, axis=1)  # (n,)
    grad_pair = -p * (c * row_sum[:, None] - inv_dist_p2 @ c)
    
    # Wall clearance
    # walls[i, 0] = HALF - c[i,0]
    # walls[i, 1] = HALF + c[i,0]
    # walls[i, 2] = HALF - c[i,1]
    # walls[i, 3] = HALF + c[i,1]
    walls = np.stack([
        HALF - c[:, 0],
        HALF + c[:, 0],
        HALF - c[:, 1],
        HALF + c[:, 1]
    ], axis=1)  # (n, 4)
    walls = np.maximum(walls, eps)
    
    wall_energy = np.sum(np.power(walls, -p))
    
    # Gradient from walls:
    # d/d c_i [ (1/w_ik^p) ] = -p * (1/w_ik^(p+1)) * d(w_ik)/d c_i
    # d(wall_0)/dc_x = -1, d(wall_1)/dc_x = +1, d(wall_2)/dc_y = -1, d(wall_3)/dc_y = +1
    inv_wall_p1 = np.power(walls, -(p + 1))  # (n, 4)
    grad_wall = np.zeros_like(c)
    # For x-component: wall_0 contributes -1, wall_1 contributes +1
    grad_wall[:, 0] = -p * (-inv_wall_p1[:, 0] + inv_wall_p1[:, 1])
    # For y-component: wall_2 contributes -1, wall_3 contributes +1
    grad_wall[:, 1] = -p * (-inv_wall_p1[:, 2] + inv_wall_p1[:, 3])
    
    total_energy = pair_energy + wall_energy
    total_grad = grad_pair + grad_wall
    
    return total_energy, total_grad.flatten()


def _optimize_from(c0, p, deadline, rng):
    """Run L-BFGS-B from c0 with given power p, respecting deadline."""
    flat0 = c0.flatten()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    t0 = time.time()
    remaining = deadline - (time.time() - t_start)
    if remaining < 5.0:
        return c0, _radius_of(c0)
    
    best_c = c0.copy()
    best_r = _radius_of(c0)
    
    try:
        res = minimize(
            _energy_and_grad,
            flat0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 5000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'iprint': -1,
                'maxls': 50,
                'maxcor': 20
            }
        )
        if res.success or res.fun < 1e10:
            c_opt = res.x.reshape(N, 2)
            c_opt = np.clip(c_opt, -HALF + 1e-9, HALF - 1e-9)
            r_opt = _radius_of(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt
    except Exception:
        pass
    
    return best_c, best_r


def _polish(c0, deadline):
    """Final sub-gradient polish on the true min-distance objective."""
    c = c0.copy().astype(float)
    best = c.copy()
    best_r = _radius_of(c)
    n = N
    t0 = time.time()
    
    for it in range(20000):
        if time.time() - t0 > deadline:
            break
        
        d = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_half = dist / 2.0
        
        walls = np.stack([
            HALF - c[:, 0], HALF + c[:, 0],
            HALF - c[:, 1], HALF + c[:, 1]
        ], axis=1)
        
        pair_min = np.min(pair_half)
        wall_min = np.min(walls)
        r = min(pair_min, wall_min)
        
        lr = 0.0001 * (1.0 - it / 20000) + 0.00001
        grad = np.zeros_like(c)
        
        # Pair contributions (all pairs within epsilon of min)
        eps = 1e-7
        mask_pair = (pair_half <= pair_min + eps) & (pair_half < np.inf)
        # Vectorized: for each i, sum over j where mask_pair[i,j]
        # grad[i] += 0.5 * sum_j mask_pair[i,j] * (c[i]-c[j])/dist[i,j]
        for i in range(n):
            js = np.where(mask_pair[i, :])[0]
            if len(js) > 0:
                diffs = (c[i] - c[js]) / np.maximum(dist[i, js], 1e-12)
                grad[i] += 0.5 * np.sum(diffs, axis=0)
                for j in js:
                    grad[j] += 0.5 * (-diffs[list(js).index(j)])
        
        # Wall contributions
        mask_wall = (walls <= wall_min + eps)
        for i in range(n):
            for k in range(4):
                if mask_wall[i, k]:
                    if k == 0:
                        grad[i, 0] -= 0.5
                    elif k == 1:
                        grad[i, 0] += 0.5
                    elif k == 2:
                        grad[i, 1] -= 0.5
                    elif k == 3:
                        grad[i, 1] += 0.5
        
        c_new = c + lr * grad
        c_new = np.clip(c_new, -HALF + 1e-10, HALF - 1e-10)
        
        r_new = _radius_of(c_new)
        if r_new > best_r:
            best_r = r_new
            best = c_new.copy()
        c = c_new
    
    return best, best_r


def construct():
    global t_start
    t_start = time.time()
    deadline = T - MARGIN  # 140s
    rng = np.random.default_rng(SEED)
    
    # Start from hexagonal seed
    c0 = _hexagon_seed(N, r0=0.0366)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)
    
    # Multi-restart with different perturbation scales and power values
    # Use increasing p for finer resolution
    p_schedule = [8, 12, 16, 20, 24, 30, 40, 50, 60, 80]
    
    for p in p_schedule:
        if time.time() - t_start > deadline - 30.0:
            break
        
        # Multiple restarts per p value if time allows
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() - t_start > deadline - 30.0:
                break
            
            # Perturb the current best
            scale = 0.01 / (1.0 + 0.2 * restart)
            c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
            c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
            
            # Ensure no coincident centres
            d = c_start[:, None, :] - c_start[None, :, :]
            dist = np.sqrt(np.sum(d * d, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 1e-5:
                c_start = best_c.copy()
            
            rem = deadline - (time.time() - t_start) - 30.0
            if rem < 5.0:
                break
            
            c_opt, r_opt = _optimize_from(c_start, p, rem, rng)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt
    
    # Final polish pass
    rem = deadline - (time.time() - t_start)
    if rem > 3.0:
        polished, r_pol = _polish(best_c, rem - 1.0)
        if r_pol > best_r:
            best_r = r_pol
            best_c = polished
    
    # Final safety
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_c


if __name__ == "__main__":
    t_start = time.time()
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
