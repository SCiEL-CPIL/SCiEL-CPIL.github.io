"""Binary search on radius with projected gradient descent for 50 circles in unit square."""
import numpy as np
import time

N = 50
HALF = 0.5


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _feasible(pts, r, max_iter=2000):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses projected gradient descent to push overlapping circles apart and
    keep them within bounds. Returns (success, pts).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    for it in range(max_iter):
        # Compute all pairwise distances and forces
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        
        # Gradient of the objective: we want to maximize the minimum distance
        # For pairs closer than 2r, push them apart
        # Force on i from j: direction (i - j) / |i - j| * (2r - dist_ij) if dist < 2r
        # We want to increase the minimum pairwise distance
        
        # Compute forces for overlapping pairs
        overlap = (dist < 2 * r) & (dist > 1e-12)
        # Scale: push apart by proportional amount
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        # Normalize directions
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        # Sum forces on each point
        grad = np.sum(force, axis=1)  # (n, 2)
        
        # Wall constraints: push away from walls
        # For each coordinate, if pts[i,k] + r > 0.5, push left; if pts[i,k] - r < -0.5, push right
        wall_force = np.zeros_like(pts)
        for k in range(2):
            too_right = pts[:, k] > HALF - r
            too_left = pts[:, k] < -HALF + r
            wall_force[too_right, k] = -(pts[too_right, k] - (HALF - r))
            wall_force[too_left, k] = (pts[too_left, k] - (-HALF + r))
        
        grad += wall_force
        
        # Adaptive step size
        step = 0.05
        # Try to move
        new_pts = pts + step * grad
        
        # Project back to bounds
        new_pts = np.clip(new_pts, -HALF + r, HALF - r)
        
        # Check if all constraints are satisfied
        min_pair = np.min(dist)
        min_wall = np.min(HALF - np.abs(new_pts))
        
        if min_pair >= 2 * r - 1e-10 and min_wall >= r - 1e-10:
            return True, new_pts
        
        pts = new_pts
        
        # Early termination if not making progress
        if it > 100:
            cur_min = min(np.min(np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))),
                          np.min(HALF - np.abs(pts)))
            if cur_min < 2 * r - 1e-8:
                # Check if we've converged
                if it > 500:
                    break
    
    # Final check
    dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    success = (min_pair >= 2 * r - 1e-10) and (min_wall >= r - 1e-10)
    return success, pts


def _hex_seed(r):
    """Generate a hexagonal lattice seed with spacing 2r."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """Generate a grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Start with a reasonable radius estimate
    # Best known is ~0.071377
    r_low = 0.06
    r_high = 0.08
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    for it in range(30):  # ~1e-9 precision after 30 bits
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try multiple seeds for this radius
        success = False
        trial_pts = None
        
        seeds = [_hex_seed(r_mid), _grid_seed(r_mid)]
        # Add some random perturbations
        for _ in range(2):
            seed = _hex_seed(r_mid) + np.random.randn(N, 2) * 0.01
            seed = np.clip(seed, -HALF + r_mid, HALF + r_mid)
            seeds.append(seed)
        
        for seed in seeds:
            if time.time() > deadline:
                break
            ok, pts = _feasible(seed, r_mid, max_iter=1000)
            if ok:
                success = True
                trial_pts = pts
                break
        
        if success:
            r_low = r_mid
            best_pts = trial_pts.copy()
            best_r = _score(best_pts)
        else:
            r_high = r_mid
        
        # If we've been successful, try to improve with the current best
        if r_low > best_r:
            best_r = r_low
    
    # Final refinement: try to squeeze a bit more
    # Use the best configuration and try local optimization
    if time.time() < deadline - 10:
        r_target = _score(best_pts) + 0.001
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            ok, pts = _feasible(best_pts, r_target, max_iter=500)
            if ok:
                best_pts = pts
                best_r = _score(best_pts)
            else:
                break
    
    return best_pts
