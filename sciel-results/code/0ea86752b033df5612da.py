"""Circle packing in a unit square: 200 equal circles, maximize common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit

def _score(c):
    """Compute the common radius for centres c of shape (N,2)."""
    # Boundary distance: min over i,k of (0.5 - |c[i,k]|)
    boundary = 0.5 - np.max(np.abs(c), axis=1)
    r_boundary = np.min(boundary)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_pair, r_boundary)

def _force_step(c, beta, dt):
    """One vectorized force-directed step: repulsion between circles + wall repulsion."""
    n = c.shape[0]
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    # Avoid division by zero
    dist2 = np.maximum(dist2, 1e-12)
    dist = np.sqrt(dist2)
    # Repulsive force: proportional to 1/dist^2 (soft repulsion)
    # F_ij = beta * diff_ij / dist_ij^3  (push apart)
    # But we want them to separate, so force on i from j is beta * (c_i - c_j) / dist^3
    # Actually standard: F = beta * diff / dist^3, where diff = c_i - c_j
    # This pushes i away from j.
    # But we need to be careful: if dist is small, force is large.
    # Use softening: F = beta * diff / (dist^2 + eps)^1.5
    eps = 1e-6
    denom = (dist2 + eps) ** 1.5
    np.fill_diagonal(denom, 1.0)  # avoid self-interaction
    force = beta * diff / denom[:, :, np.newaxis]
    # Sum forces on each circle
    net_force = np.sum(force, axis=1)
    # Wall repulsion: push circles away from walls
    # For each circle, if it's close to a wall, add force pushing inward
    # Wall at x = -0.5: force in +x direction if x < -0.5 + margin
    # Use a quadratic penalty near walls
    wall_margin = 0.01
    # x-coordinates
    x = c[:, 0]
    y = c[:, 1]
    # Left wall (x = -0.5): push right if x < -0.5 + wall_margin
    fx_wall = np.zeros(n)
    fy_wall = np.zeros(n)
    # Left wall
    mask = x < -0.5 + wall_margin
    if np.any(mask):
        fx_wall[mask] += beta * (wall_margin - (x[mask] + 0.5)) ** 2
    # Right wall
    mask = x > 0.5 - wall_margin
    if np.any(mask):
        fx_wall[mask] -= beta * (x[mask] - (0.5 - wall_margin)) ** 2
    # Bottom wall
    mask = y < -0.5 + wall_margin
    if np.any(mask):
        fy_wall[mask] += beta * (wall_margin - (y[mask] + 0.5)) ** 2
    # Top wall
    mask = y > 0.5 - wall_margin
    if np.any(mask):
        fy_wall[mask] -= beta * (y[mask] - (0.5 - wall_margin)) ** 2
    net_force[:, 0] += fx_wall
    net_force[:, 1] += fy_wall
    # Update positions
    c_new = c + dt * net_force
    # Clip to stay within square (with small buffer)
    c_new = np.clip(c_new, -0.5 + 1e-10, 0.5 - 1e-10)
    return c_new

def _lbfgs_polish(c, maxiter=2000, tol=1e-15):
    """Polish using L-BFGS-B to maximize the minimum radius."""
    def neg_radius(c_flat):
        c = c_flat.reshape(-1, 2)
        return -_score(c)
    
    def neg_radius_grad(c_flat):
        c = c_flat.reshape(-1, 2)
        n = c.shape[0]
        # Compute gradient of -min(distances)
        # This is complex; use finite differences as fallback if analytic is too slow
        # For now, let scipy use finite differences (slow but accurate)
        return None  # Let scipy compute numerically
    
    # Use a simpler approach: maximize the soft-min approximation
    # Or just use L-BFGS-B with numerical gradient on the objective
    
    res = minimize(
        neg_radius,
        c.ravel(),
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    c_opt = res.x.reshape(-1, 2)
    # Clip to valid range
    c_opt = np.clip(c_opt, -0.5 + 1e-10, 0.5 - 1e-10)
    return c_opt

def _make_hex_initial():
    """Generate a hexagonal close-packed initial configuration."""
    # Hexagonal packing: rows offset by half the horizontal spacing
    # For N=200, use roughly 15x14 or similar arrangement
    # Let's use a hex grid and take the first N points
    pts = []
    row = 0
    while len(pts) < N:
        col = 0
        while len(pts) < N:
            x = col * 1.0  # will scale later
            y = row * 0.8660254037844386  # sqrt(3)/2
            if row % 2 == 1:
                x += 0.5
            pts.append([x, y])
            col += 1
        row += 1
    pts = np.array(pts[:N])
    # Normalize to fit in [-0.5, 0.5]^2
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    scale = min(0.9 / (xmax - xmin), 0.9 / (ymax - ymin)) if (xmax > xmin and ymax > ymin) else 1.0
    pts = (pts - [xmin, ymin]) * scale
    pts += [xmin * scale - 0.5 + (0.9 - (xmax - xmin) * scale) / 2,
             ymin * scale - 0.5 + (0.9 - (ymax - ymin) * scale) / 2]
    # Simpler: center and scale
    pts = (pts - pts.mean(axis=0))
    max_extent = np.max(np.abs(pts))
    if max_extent > 0:
        pts *= 0.45 / max_extent
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial configuration: hexagonal packing
    c = _make_hex_initial()
    best_c = c.copy()
    best_r = _score(c)
    
    # Phase 1: Force-directed relaxation
    # Start with moderate beta, increase over time
    n_iters_phase1 = 5000
    dt = 0.001
    beta = 1.0
    
    for i in range(n_iters_phase1):
        if time.time() > deadline:
            break
        beta = 1.0 + 4000.0 * i / n_iters_phase1  # ramp up to ~4000
        c = _force_step(c, beta, dt)
        r = _score(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Phase 2: L-BFGS-B polish
    if time.time() < deadline - 30:
        c = _lbfgs_polish(best_c, maxiter=3000, tol=1e-15)
        r = _score(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Phase 3: Another force-directed pass with higher beta for final sharpening
    if time.time() < deadline - 20:
        c = best_c.copy()
        n_iters_phase3 = 2000
        dt = 0.0005
        for i in range(n_iters_phase3):
            if time.time() > deadline:
                break
            beta = 5000.0 + 10000.0 * i / n_iters_phase3
            c = _force_step(c, beta, dt)
            r = _score(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    # Final L-BFGS-B polish
    if time.time() < deadline - 10:
        c = _lbfgs_polish(best_c, maxiter=2000, tol=1e-15)
        r = _score(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Ensure validity: clip to square, check no coincident points
    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    # Final validation
    if not np.all(np.isfinite(best_c)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
