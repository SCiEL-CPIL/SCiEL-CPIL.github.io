"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 125  # safety margin under 150s kill


def _radius(pts):
    """Compute the common radius for a set of circle centres."""
    n = len(pts)
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # distance to walls (square [-0.5, 0.5]^2)
    wall = np.minimum(0.5 - np.abs(pts), 0.5)
    min_wall = wall.min()
    return min(min_pair, 2.0 * min_wall) / 2.0


def _energy(pts, alpha=8):
    """
    Smooth inverse-power-sum energy objective.
    Minimizing this encourages maximizing the minimum distance.
    """
    p = pts
    # Pairwise distances
    diff = p[:, None, :] - p[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)  # avoid division by zero on diagonal
    
    # Inverse power sum for pairs
    pair_energy = np.sum(1.0 / (d2 ** (alpha / 2.0)))
    
    # Wall energy: penalize getting close to walls
    wall_dist = np.minimum(0.5 - np.abs(p), 0.5)
    wall_energy = np.sum(1.0 / (np.maximum(wall_dist, 1e-8) ** alpha))
    
    return pair_energy + wall_energy


def _grad_energy(pts, alpha=8):
    """
    Gradient of the smooth energy objective.
    """
    p = pts
    n = len(p)
    grad = np.zeros_like(p)
    
    # Pairwise gradient
    diff = p[:, None, :] - p[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    
    # d/dp_i sum_{j!=i} 1/d_ij^alpha = -alpha * sum_{j!=i} (p_i - p_j) / d_ij^(alpha+2)
    inv_d2_pow = (1.0 / (d2 ** ((alpha + 2) / 2.0)))
    np.fill_diagonal(inv_d2_pow, 0.0)
    
    # For each i: grad_i = -alpha * sum_j (p_i - p_j) * inv_d2_pow[i,j]
    # = -alpha * (sum_j p_i * inv_d2_pow[i,j] - sum_j p_j * inv_d2_pow[i,j])
    # = -alpha * (p_i * sum_j inv_d2_pow[i,j] - sum_j p_j * inv_d2_pow[i,j])
    
    row_sum = inv_d2_pow.sum(axis=1, keepdims=True)  # (n, 1)
    weighted_sum = inv_d2_pow @ p  # (n, 2)
    
    grad_pair = -alpha * (p * row_sum - weighted_sum)
    
    # Wall gradient
    wall_dist = np.minimum(0.5 - np.abs(p), 0.5)
    wall_dist_safe = np.maximum(wall_dist, 1e-8)
    # d/dx (1/wall_dist^alpha) where wall_dist = 0.5 - |x|
    # For x > 0: wall_dist = 0.5 - x, d/dx = -1, so d/dx (1/wd^alpha) = alpha / wd^(alpha+1)
    # For x < 0: wall_dist = 0.5 + x, d/dx = +1, so d/dx (1/wd^alpha) = -alpha / wd^(alpha+1)
    # Combined: d/dx = -sign(x) * alpha / wd^(alpha+1)
    sign = np.sign(p)
    sign[sign == 0] = 1.0  # at x=0, either sign works, use +1
    grad_wall = -sign * alpha / (wall_dist_safe ** (alpha + 1))
    
    return grad_pair + grad_wall


def _hex_initial(n):
    """Generate a hexagonal lattice initial placement inside [-0.5,0.5]^2."""
    pts = []
    # hex spacing
    dx = 1.0 / (np.sqrt(3.0) * 14)  # roughly 14 columns
    dy = dx * np.sqrt(3.0) / 2.0
    x0, y0 = -0.5 + dx, -0.5 + dy
    row = 0
    while len(pts) < n:
        offset = (row % 2) * dx / 2.0
        x = x0 + offset
        while x < 0.5 - dx / 2:
            y = y0 + row * dy
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                pts.append([x, y])
            x += dx
            if len(pts) >= n:
                break
        row += 1
    return np.array(pts[:n])


def _distorted_hex_initial(n):
    """
    Generate a distorted hexagonal lattice using a bilinear warp
    to better fit the square boundary.
    """
    # Start with ideal hexagonal points
    base = _hex_initial(n)
    
    # Apply a smooth distortion that pushes points away from boundaries
    # This is a simple radial distortion from the center
    cx, cy = 0.0, 0.0
    r = np.sqrt((base[:, 0] - cx)**2 + (base[:, 1] - cy)**2)
    r_max = 0.707  # half diagonal of unit square
    
    # Smoothly compress points near the boundary
    t = np.clip(r / r_max, 0, 1)
    # Use a smoothstep-like function
    warp = 1.0 - 0.15 * (3*t**2 - 2*t**3)  # gentle inward pull
    
    base[:, 0] = cx + (base[:, 0] - cx) * warp
    base[:, 1] = cy + (base[:, 1] - cy) * warp
    
    # Ensure points are inside the square
    base = np.clip(base, -0.495, 0.495)
    
    return base


def _grid_initial(n):
    """Square grid initial placement."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:n]


def _optimize(pts0, max_time=110.0, alpha=8):
    """Run L-BFGS-B optimization with smooth energy objective."""
    start = time.time()
    flat = pts0.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)

    def fun(x):
        if time.time() - start > max_time:
            raise ValueError("timeout")
        return _energy(x.reshape(N, 2), alpha)

    def jac(x):
        if time.time() - start > max_time:
            raise ValueError("timeout")
        return _grad_energy(x.reshape(N, 2), alpha).flatten()

    result = None
    try:
        result = minimize(
            fun, flat, jac=jac, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < fun(flat):
            flat = result.x
    except (ValueError, Exception):
        pass

    # Check time
    elapsed = time.time() - start
    if elapsed < max_time - 15:
        # Second round with tighter tolerance
        try:
            result2 = minimize(
                fun, flat, jac=jac, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.fun < fun(flat):
                flat = result2.x
        except (ValueError, Exception):
            pass

    return flat.reshape(N, 2)


def construct():
    t0 = time.time()
    best_pts = None
    best_r = -1.0

    initializers = [
        _hex_initial(N),
        _distorted_hex_initial(N),
        _grid_initial(N),
    ]

    # Add a few perturbed versions of hex
    rng = np.random.RandomState(42)
    for _ in range(2):
        base = _hex_initial(N).copy()
        noise = rng.normal(0, 0.008, size=base.shape)
        base = np.clip(base + noise, -0.49, 0.49)
        initializers.append(base)

    for init in initializers:
        if time.time() - t0 > 115:
            break
        try:
            remaining = min(110.0, 115.0 - (time.time() - t0))
            if remaining < 10:
                break
            pts = _optimize(init, max_time=remaining, alpha=8)
            r = _radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts
        except Exception:
            continue

    if best_pts is None:
        best_pts = _hex_initial(N)

    # Final time check
    if time.time() - t0 > 130:
        return best_pts

    return best_pts
# EVOLVE-BLOCK-END
