"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time

DEADLINE = 150.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - _elapsed() - 8.0  # 8s safety margin


def _max_radius(x, y, centers, radii, idx):
    """Maximum radius for a circle at (x,y) given all other circles and boundaries."""
    r_max = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(len(centers)):
        if j == idx:
            continue
        dx = x - centers[j, 0]
        dy = y - centers[j, 1]
        d = np.sqrt(dx * dx + dy * dy)
        r_max = min(r_max, d - radii[j])
    return max(0.0, r_max)


def _grow_all(centers, radii):
    """Greedily grow each circle to max feasible radius. Repeat until stable."""
    n = len(radii)
    for _ in range(20):
        changed = False
        for i in range(n):
            new_r = _max_radius(centers[i, 0], centers[i, 1], centers, radii, i)
            if new_r > radii[i] + 1e-10:
                radii[i] = new_r
                changed = True
        if not changed:
            break
    return centers, radii


def _local_optimal_center(x, y, centers, radii, idx):
    """
    Find the local optimum position for circle idx given all other circles.
    The optimum is where the gradient of the clearance function is zero.
    We use a small Nelder-Mead-like search around the current position.
    """
    n = len(centers)
    best_x, best_y = x, y
    best_r = _max_radius(x, y, centers, radii, idx)
    
    # Multi-scale local search
    scales = [0.05, 0.02, 0.01, 0.005, 0.002]
    for scale in scales:
        improved = True
        while improved and _remaining() > 5.0:
            improved = False
            # Try 8 directions + center
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = best_x + dx
                    ny = best_y + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    r = _max_radius(nx, ny, centers, radii, idx)
                    if r > best_r + 1e-8:
                        best_x, best_y = nx, ny
                        best_r = r
                        improved = True
    return best_x, best_y, best_r


def _recenter_and_grow(centers, radii, iterations=30):
    """Iteratively re-center each circle to local optimum and regrow."""
    n = len(radii)
    for it in range(iterations):
        if _remaining() < 10.0:
            break
        total_r_before = np.sum(radii)
        for i in range(n):
            # Find better center for circle i
            new_x, new_y, new_r = _local_optimal_center(
                centers[i, 0], centers[i, 1], centers, radii, i
            )
            centers[i, 0] = new_x
            centers[i, 1] = new_y
            radii[i] = new_r
        # Regrow all
        centers, radii = _grow_all(centers, radii)
        total_r_after = np.sum(radii)
        if total_r_after - total_r_before < 1e-10:
            break
    return centers, radii


def _is_valid(centers, radii, tol=1e-5):
    """Check if packing is valid."""
    n = len(radii)
    for i in range(n):
        if radii[i] < -tol:
            return False
        if centers[i, 0] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 0] - radii[i] < -tol:
            return False
        if centers[i, 1] - radii[i] < -tol:
            return False
        if 1.0 - centers[i, 1] - radii[i] < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                        (centers[i, 1] - centers[j, 1])**2)
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _fix_packing(centers, radii):
    """Fix small violations."""
    centers = centers.copy()
    radii = radii.copy()
    for i in range(len(radii)):
        centers[i, 0] = np.clip(centers[i, 0], 0.0, 1.0)
        centers[i, 1] = np.clip(centers[i, 1], 0.0, 1.0)
    for i in range(len(radii)):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max_r)
        radii[i] = max(0.0, radii[i])
    for _ in range(100):
        changed = False
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                            (centers[i, 1] - centers[j, 1])**2)
                needed = radii[i] + radii[j]
                if d < needed:
                    excess = needed - d
                    # Shrink proportionally
                    if radii[i] + radii[j] > 1e-10:
                        frac_i = radii[i] / (radii[i] + radii[j])
                        frac_j = radii[j] / (radii[i] + radii[j])
                    else:
                        frac_i = frac_j = 0.5
                    radii[i] = max(0.0, radii[i] - excess * frac_i * 1.01)
                    radii[j] = max(0.0, radii[j] - excess * frac_j * 1.01)
                    changed = True
        if not changed:
            break
    return centers, radii


def _perturb(centers, radii, amount=0.02):
    """Perturb centers slightly to escape local minima."""
    centers_p = centers.copy()
    radii_p = radii.copy()
    n = len(radii)
    # Perturb a random subset
    idxs = np.random.choice(n, size=max(1, n // 3), replace=False)
    for i in idxs:
        centers_p[i, 0] += np.random.uniform(-amount, amount)
        centers_p[i, 1] += np.random.uniform(-amount, amount)
        centers_p[i, 0] = np.clip(centers_p[i, 0], 0.02, 0.98)
        centers_p[i, 1] = np.clip(centers_p[i, 1], 0.02, 0.98)
    # Regrow
    centers_p, radii_p = _grow_all(centers_p, radii_p)
    return centers_p, radii_p


def _hex_init(n, cols, rows, jitter=0.0):
    """Hexagonal packing initialization."""
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * dx + (0.5 * dx if r % 2 == 1 else 0)
            y = (r + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    centers = centers[:n]
    return np.array(centers)


def _grid_init(n, cols, rows, jitter=0.0):
    """Grid initialization."""
    dx = 1.0 / cols
    dy = 1.0 / rows
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * dx
            y = (r + 0.5) * dy
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            centers.append((x, y))
    centers = centers[:n]
    return np.array(centers)


def _random_init(n, seed=None):
    """Random initialization with minimum spacing."""
    rng = np.random.RandomState(seed)
    centers = []
    min_dist = 0.08
    attempts = 0
    while len(centers) < n and attempts < 20000:
        attempts += 1
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        ok = True
        for cx, cy in centers:
            if (x - cx)**2 + (y - cy)**2 < min_dist**2:
                ok = False
                break
        if ok:
            centers.append((x, y))
    while len(centers) < n:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        centers.append((x, y))
    return np.array(centers)


def _optimize_from_start(centers_init, radii_init):
    """Full optimization pipeline from a starting configuration."""
    if _remaining() < 15.0:
        return centers_init, radii_init, np.sum(radii_init)
    
    centers = centers_init.copy()
    radii = radii_init.copy()
    
    # Phase 1: Grow all
    centers, radii = _grow_all(centers, radii)
    
    # Phase 2: Re-center and regrow
    centers, radii = _recenter_and_grow(centers, radii, iterations=25)
    
    # Phase 3: Perturb and re-optimize (multiple rounds)
    best_sum = np.sum(radii)
    best_c, best_r = centers.copy(), radii.copy()
    
    for round_idx in range(5):
        if _remaining() < 10.0:
            break
        # Perturb
        c_p, r_p = _perturb(centers, radii, amount=0.03 - round_idx * 0.005)
        # Re-optimize
        c_p, r_p = _grow_all(c_p, r_p)
        c_p, r_p = _recenter_and_grow(c_p, r_p, iterations=15)
        
        if _is_valid(c_p, r_p, tol=1e-5):
            s = np.sum(r_p)
            if s > best_sum + 1e-8:
                best_sum = s
                best_c, best_r = c_p.copy(), r_p.copy()
                centers, radii = c_p, r_p
    return best_c, best_r, best_sum


def construct():
    n = N
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal 6x5
    for jitter in [0.0, 0.01, 0.02]:
        c = _hex_init(n, 6, 5, jitter=jitter)
        r = np.array([min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1]) * 0.5 for i in range(n)])
        inits.append((c, r))
    
    # Hexagonal 5x6
    for jitter in [0.0, 0.01]:
        c = _hex_init(n, 5, 6, jitter=jitter)
        r = np.array([min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1]) * 0.5 for i in range(n)])
        inits.append((c, r))
    
    # Grid 5x6
    for jitter in [0.0, 0.01]:
        c = _grid_init(n, 5, 6, jitter=jitter)
        r = np.array([min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1]) * 0.5 for i in range(n)])
        inits.append((c, r))
    
    # Grid 6x5
    c = _grid_init(n, 6, 5, jitter=0.0)
    r = np.array([min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1]) * 0.5 for i in range(n)])
    inits.append((c, r))
    
    # Random starts
    for seed in [42, 123, 7, 999]:
        c = _random_init(n, seed=seed)
        r = np.array([min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1]) * 0.4 for i in range(n)])
        inits.append((c, r))
    
    # Optimize from each start
    for idx, (c_init, r_init) in enumerate(inits):
        if _remaining() < 12.0:
            break
        
        centers, radii, total = _optimize_from_start(c_init, r_init)
        
        if _is_valid(centers, radii, tol=1e-5):
            if total > best_sum:
                best_sum = total
                best_centers = centers
                best_radii = radii
        else:
            centers_f, radii_f = _fix_packing(centers, radii)
            if _is_valid(centers_f, radii_f, tol=1e-5):
                total_f = np.sum(radii_f)
                if total_f > best_sum:
                    best_sum = total_f
                    best_centers = centers_f
                    best_radii = radii_f
    
    # Final polish: one more recenter pass on the best
    if best_centers is not None and _remaining() > 10.0:
        best_centers, best_radii = _recenter_and_grow(best_centers, best_radii, iterations=10)
        if not _is_valid(best_centers, best_radii, tol=1e-5):
            best_centers, best_radii = _fix_packing(best_centers, best_radii)
        best_sum = np.sum(best_radii)
    
    # Fallback
    if best_centers is None:
        side = int(np.ceil(np.sqrt(n)))
        dx = 1.0 / side
        dy = 1.0 / side
        centers = []
        radii = []
        for j in range(side):
            for i in range(side):
                if len(centers) >= n:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                r = 0.5 * dx - 1e-3
                centers.append((x, y))
                radii.append(r)
        best_centers = np.array(centers[:n])
        best_radii = np.array(radii[:n])
    
    result = np.column_stack([best_centers, best_radii])
    return result

# EVOLVE-BLOCK-END
