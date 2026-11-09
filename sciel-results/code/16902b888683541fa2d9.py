"""Model-and-optimize: hexagonal seed + L-BFGS-B on soft-min distance with deadline guard."""
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
    # Use a denser hex grid and pick the best subset
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


def _soft_min_obj(x, beta=50.0):
    """
    Soft-min of distances: -softmin(min pairwise dist/2, wall clearance).
    We maximize this, so minimize the negative.
    x is flat (2N,) array of centers.
    """
    c = x.reshape(-1, 2)
    # Wall clearance
    wall = HALF - np.abs(c)
    min_wall = np.min(wall)
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Soft-min of the two constraints
    # We want to maximize min(min_wall, min_pair)
    # Use soft-min: softmin(a,b) = -log(exp(-beta*a)+exp(-beta*b))/beta
    # But for numerical stability with large beta, use a simpler approach
    # Actually, let's just use the hard min with a smooth penalty for violations
    
    # Better: use a smooth approximation of min
    # min(a,b) ≈ -(1/beta)*log(exp(-beta*a) + exp(-beta*b))
    # But this can overflow. Use the log-sum-exp trick.
    
    a = min_wall
    b = min_pair
    m = np.minimum(a, b)
    if a - m < 1e-10 and b - m < 1e-10:
        soft = m
    else:
        # Stable soft-min
        t = np.maximum(a, b)
        soft = t + np.log(np.exp(-beta * (a - t)) + np.exp(-beta * (b - t))) / beta
    
    return -soft


def _soft_min_grad(x, beta=50.0):
    """Gradient of soft-min objective."""
    c = x.reshape(-1, 2)
    # Wall clearance
    wall = HALF - np.abs(c)
    min_wall = np.min(wall)
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    a = min_wall
    b = min_pair
    m = np.minimum(a, b)
    
    # Soft-min gradient
    if a - m < 1e-10 and b - m < 1e-10:
        # Both are at the minimum, split gradient evenly
        ea = 0.5
        eb = 0.5
    else:
        t = np.maximum(a, b)
        ea = np.exp(-beta * (a - t))
        eb = np.exp(-beta * (b - t))
        s = ea + eb
        ea /= s
        eb /= s
    
    grad = np.zeros_like(x)
    
    # Gradient of wall clearance: d/dx (HALF - |x|) = -sign(x)
    # d(min_wall)/dx_i = -sign(c_i) for the argmin coordinate
    # But we need the full gradient of the soft-min
    
    # Actually, computing exact gradients is complex. Let's use numerical gradient
    # or just use finite differences in scipy.
    
    # For simplicity, return None to let scipy use numerical gradient
    return None


def construct():
    t_start = time.time()
    deadline = 110.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    beta = 80.0  # soft-min sharpness

    restarts = 0
    while time.time() - t_start < deadline:
        restarts += 1
        # Perturbation magnitude shrinks over restarts
        scale = 0.015 / (1.0 + 0.15 * restarts)
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
        bounds = [(-HALF + 1e-6, HALF - 1e-6)] * (2 * N)

        try:
            result = minimize(
                _soft_min_obj,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if result.success or result.fun < _soft_min_obj(x0, beta):
                c_opt = result.x.reshape(-1, 2)
                c_opt = np.clip(c_opt, -HALF + 1e-6, HALF - 1e-6)
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
    return best_c


if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
