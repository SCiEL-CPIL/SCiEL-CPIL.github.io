"""Circle packing in a perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np

N = 21
_DEADLINE = 110.0  # safety margin below 120s limit

def _init_grid(W, H):
    """5x5 grid (20) + 1 center = 21. Positions and radii."""
    pos = []
    rad = []
    for i in range(5):
        for j in range(5):
            pos.append((i + 0.5) * W / 5.0)
            pos.append((j + 0.5) * H / 5.0)
            rad.append(min(W, H) / 10.0)
    pos.append(W / 2.0)
    pos.append(H / 2.0)
    rad.append(min(W, H) / 20.0)
    pos = np.array(pos)
    rad = np.array(rad)
    return pos, rad

def _objective(pos, rad, W, H):
    """Larger value = better (closer to feasible + larger radii)."""
    # Penalties for constraint violations
    pen = 0.0
    # Inside rectangle
    for k in range(N):
        x, y, r = pos[2*k], pos[2*k+1], rad[k]
        if x < r:
            pen += (r - x) ** 2
        if x > W - r:
            pen += (x - (W - r)) ** 2
        if y < r:
            pen += (r - y) ** 2
        if y > H - r:
            pen += (y - (H - r)) ** 2
    # Non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[2*i] - pos[2*j]
            dy = pos[2*i+1] - pos[2*j+1]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = rad[i] + rad[j]
            if d < min_d:
                pen += (min_d - d) ** 2
    return -pen + np.sum(rad)

def _random_search(pos0, rad0, W, H, budget):
    """Random perturbation search with radius growth."""
    best_val = _objective(pos0, rad0, W, H)
    best_pos, best_rad = pos0.copy(), rad0.copy()
    t0 = time.time()
    scale = 0.05
    while time.time() - t0 < budget:
        # Random perturbation
        dpos = np.random.normal(0, scale, size=2*N)
        dpos[0::2] *= W
        dpos[1::2] *= H
        drad = np.random.normal(0, 0.005, size=N)
        new_pos = best_pos + dpos
        new_rad = best_rad + drad
        new_rad = np.clip(new_rad, 1e-4, min(W, H) * 0.5)
        val = _objective(new_pos, new_rad, W, H)
        if val > best_val:
            best_val = val
            best_pos, best_rad = new_pos, new_rad
        # Anneal scale
        scale *= 0.999
    return best_pos, best_rad, best_val

def _optimize(W, H, budget):
    """Optimize for a given rectangle aspect ratio."""
    pos, rad = _init_grid(W, H)
    pos, rad, val = _random_search(pos, rad, W, H, budget)
    return pos, rad, val

def _is_valid(pos, rad, W, H, tol=1e-6):
    """Check all constraints with tolerance."""
    for k in range(N):
        x, y, r = pos[2*k], pos[2*k+1], rad[k]
        if r < -tol:
            return False
        if x < r - tol or x > W - r + tol or y < r - tol or y > H - r + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[2*i] - pos[2*j]
            dy = pos[2*i+1] - pos[2*j+1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < rad[i] + rad[j] - tol:
                return False
    return True

def _compute_value(pos, rad):
    """Compute objective value (sum of radii) and validate."""
    # Find W, H from the bounding box of circles
    W = 0.0
    H = 0.0
    for k in range(N):
        x, y, r = pos[2*k], pos[2*k+1], rad[k]
        W = max(W, 2*x, 2*(W - x))  # ensure circle fits
        H = max(H, 2*y, 2*(H - y))
    # Actually, the rectangle is determined by the evaluator.
    # The value is just sum of radii if valid.
    return np.sum(rad)

def construct():
    t_start = time.time()
    best_val = 0.0
    best_result = None

    # Try several aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5]
    # W + H = 2, so W = 2*r/(1+r) where r = W/H
    for ar in aspect_ratios:
        if time.time() - t_start > _DEADLINE:
            break
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        budget = min(10.0, _DEADLINE - (time.time() - t_start))
        if budget < 2.0:
            break
        pos, rad, val = _optimize(W, H, budget)
        if _is_valid(pos, rad, W, H):
            # Build result: (x, y, r) for each circle
            result = np.zeros((N, 3))
            for k in range(N):
                result[k, 0] = pos[2*k]
                result[k, 1] = pos[2*k+1]
                result[k, 2] = max(rad[k], 1e-4)
            v = np.sum(result[:, 2])
            if v > best_val:
                best_val = v
                best_result = result

    if best_result is not None:
        return best_result

    # Fallback: simple grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    # Add center circle
    cells.append([0.5, 0.5, 0.1])
    return np.array(cells[:N])

BET: {mechanism: "Optimized grid placement with per-aspect-ratio search finds tighter packing than the naive uniform grid, increasing the sum of radii by exploiting the rectangle shape and allowing circles to shift from grid centers", q: 0.75, delta: [0.005, 0.05]}
