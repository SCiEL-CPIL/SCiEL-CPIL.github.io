"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 26

# ---- helpers ----

def _staggered_init(n, scale=1.0):
    """Staggered hexagonal grid initial positions + radius."""
    # Choose a grid layout that fits n circles
    # 5 columns x 5 rows = 25, add 1 -> use 5x5 + 1, or 6x5=30
    # We'll use a 6x5 grid and pick 26
    cols, rows = 6, 5
    dx = 1.0 / cols
    dy = 1.0 / rows
    positions = []
    radii = []
    for j in range(rows):
        for i in range(cols):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            # stagger odd rows
            if j % 2 == 1:
                x += dx * 0.5
            # keep inside [0,1]
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            positions.append([x, y])
            # initial radius: half the smaller spacing
            r = 0.5 * min(dx, dy) * 0.95
            radii.append(r)
    # pick first n
    positions = np.array(positions[:n])
    radii = np.array(radii[:n])
    return positions, radii


def _force_relax(positions, radii, iters=2000, dt=0.01):
    """Simple force-directed relaxation: circles repel each other and are pushed
    toward the walls (to maximize radii)."""
    n = len(positions)
    pos = positions.copy()
    rad = radii.copy()
    for it in range(iters):
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    d = np.array([0.01, 0.01])
                    dist = np.linalg.norm(d)
                overlap = rad[i] + rad[j] - dist
                if overlap > 0:
                    force = overlap * 0.5
                    pos[i] += force * d / dist
                    pos[j] -= force * d / dist
        # Push toward walls to allow larger radii:
        # For each circle, push it toward the nearest wall
        for i in range(n):
            # distance to each wall
            d_left = pos[i, 0]
            d_right = 1.0 - pos[i, 0]
            d_bottom = pos[i, 1]
            d_top = 1.0 - pos[i, 1]
            min_d = min(d_left, d_right, d_bottom, d_top)
            # push toward nearest wall
            if min_d == d_left:
                pos[i, 0] -= 0.001
            elif min_d == d_right:
                pos[i, 0] += 0.001
            elif min_d == d_bottom:
                pos[i, 1] -= 0.001
            else:
                pos[i, 1] += 0.001
        # Keep inside [0,1]
        pos = np.clip(pos, 0.0, 1.0)
    return pos, rad


def _soft_penalty(x, n, temperature=0.01):
    """
    x: array of shape (n, 3) with (x, y, r) for each circle.
    Returns a smooth penalty for constraint violations.
    Constraints:
      1. Circle-circle: dist_ij >= r_i + r_j
      2. Wall: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    Uses soft-min (logsumexp) for smoothness.
    """
    pos = x[:, :2]
    rad = x[:, 2]
    penalty = 0.0

    # Circle-circle overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            d = pos[i] - pos[j]
            dist = np.sqrt(np.sum(d*d))
            required = rad[i] + rad[j]
            # violation = required - dist (positive if overlapping)
            viol = required - dist
            # soft penalty: use softplus-like
            penalty += np.log1p(np.exp(viol / temperature)) * temperature

    # Wall penalty: for each circle, r_i must be <= distance to each wall
    for i in range(n):
        # distance to each wall
        d_walls = np.array([pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1]])
        # min distance to any wall
        # violation = r_i - min(d_walls)
        # Use soft-min for smoothness: -logsumexp(-d_walls/temperature)*temperature
        soft_min = -logsumexp(-d_walls / temperature) * temperature
        viol = rad[i] - soft_min
        penalty += np.log1p(np.exp(viol / temperature)) * temperature

    return penalty


def _objective(x, n, temperature=0.01):
    """
    Objective to MINIMIZE: -sum(radii) + penalty
    """
    rad = x[:, 2]
    return -np.sum(rad) + _soft_penalty(x, n, temperature)


def _objective_grad(x, n, temperature=0.01):
    """
    Numerical gradient (L-BFGS-B can use numerical, but let's try analytic-ish).
    For simplicity, use finite differences with small step.
    """
    # Use central differences
    eps = 1e-7
    m = len(x)
    grad = np.zeros_like(x)
    f0 = _objective(x, n, temperature)
    for k in range(m):
        xp = x.copy()
        xp[k] += eps
        xm = x.copy()
        xm[k] -= eps
        grad[k] = (_objective(xp, n, temperature) - _objective(xm, n, temperature)) / (2*eps)
    return grad


def _is_feasible(x, n, tol=1e-6):
    """Check if the configuration is feasible (no overlaps, all inside)."""
    pos = x[:, :2]
    rad = x[:, 2]
    # Wall check
    for i in range(n):
        if rad[i] > pos[i,0] + tol or rad[i] > 1.0 - pos[i,0] + tol:
            return False
        if rad[i] > pos[i,1] + tol or rad[i] > 1.0 - pos[i,1] + tol:
            return False
    # Circle-circle check
    for i in range(n):
        for j in range(i+1, n):
            d = pos[i] - pos[j]
            dist = np.sqrt(np.sum(d*d))
            if dist < rad[i] + rad[j] - tol:
                return False
    return True


def _repair(x, n, iters=500):
    """Repair small violations by shrinking radii and nudging positions."""
    x = x.copy()
    for _ in range(iters):
        changed = False
        # Check and fix circle-circle overlaps
        for i in range(n):
            for j in range(i+1, n):
                d = x[i,:2] - x[j,:2]
                dist = np.sqrt(np.sum(d*d))
                required = x[i,2] + x[j,2]
                if dist < required - 1e-9:
                    # shrink both radii slightly
                    excess = required - dist
                    # reduce each radius by half the excess
                    shrink = excess * 0.5 + 1e-8
                    x[i,2] = max(0.0, x[i,2] - shrink)
                    x[j,2] = max(0.0, x[j,2] - shrink)
                    changed = True
        # Check and fix wall violations
        for i in range(n):
            max_r = min(x[i,0], 1.0-x[i,0], x[i,1], 1.0-x[i,1])
            if x[i,2] > max_r - 1e-9:
                x[i,2] = max(0.0, max_r - 1e-8)
                changed = True
        if not changed:
            break
    return x


def _optimize_from(x0, n, deadline, maxiter=2000, temperature=0.01):
    """Run L-BFGS-B from x0, return best feasible result found."""
    best_x = x0.copy()
    best_val = np.sum(x0[:, 2])

    # Try a few temperature values
    for temp in [0.05, 0.02, 0.01]:
        if time.time() > deadline:
            break
        x0_opt = x0.copy()
        # An annealing-like approach: start with higher temperature, decrease
        try:
            res = minimize(
                lambda x: _objective(x, n, temp),
                x0_opt,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            x_new = res.x
            # Repair
            x_new = _repair(x_new, n)
            val = np.sum(x_new[:, 2])
            if val > best_val:
                best_val = val
                best_x = x_new.copy()
        except Exception:
            pass

    return best_x


def construct():
    deadline = time.time() + 115.0  # safety margin
    best_x = None
    best_val = -1.0

    # ---- Phase 1: Generate diverse initial layouts ----
    inits = []

    # Init 1: staggered grid
    pos, rad = _staggered_init(N)
    pos, rad = _force_relax(pos, rad, iters=500, dt=0.01)
    x0 = np.column_stack([pos, rad])
    inits.append(x0)

    # Init 2: random with relaxation
    np.random.seed(42)
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        pos = np.random.uniform(0.1, 0.9, (N, 2))
        rad = np.full(N, 0.08)
        pos, rad = _force_relax(pos, rad, iters=1000, dt=0.01)
        x0 = np.column_stack([pos, rad])
        inits.append(x0)

    # Init 3: grid with different aspect
    # 5x5 grid + 1 extra
    for cols, rows in [(5, 5), (7, 4), (4, 7)]:
        if time.time() > deadline:
            break
        dx = 1.0 / cols
        dy = 1.0 / rows
        pos = []
        rad = []
        for j in range(rows):
            for i in range(cols):
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, 1.0)
                pos.append([x, y])
                rad.append(0.5 * min(dx, dy) * 0.95)
        pos = np.array(pos[:N])
        rad = np.array(rad[:N])
        pos, rad = _force_relax(pos, rad, iters=500, dt=0.01)
        x0 = np.column_stack([pos, rad])
        inits.append(x0)

    # ---- Phase 2: Optimize each initial layout ----
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
        x_opt = _optimize_from(x0, N, deadline, maxiter=1500)
        val = np.sum(x_opt[:, 2])
        if val > best_val:
            best_val = val
            best_x = x_opt.copy()

    # ---- Phase 3: Final high-precision polish ----
    if best_x is not None and time.time() < deadline:
        # Use a very low temperature for precise optimization
        try:
            res = minimize(
                lambda x: _objective(x, N, 0.005),
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            x_polish = _repair(res.x, N)
            val = np.sum(x_polish[:, 2])
            if val > best_val:
                best_val = val
                best_x = x_polish.copy()
        except Exception:
            pass

    # ---- Phase 4: Final repair to ensure feasibility ----
    if best_x is not None:
        best_x = _repair(best_x, N, iters=1000)
        # Ensure radii are non-negative
        best_x[:, 2] = np.maximum(best_x[:, 2], 0.0)

    if best_x is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_x = np.array(cells[:N])

    return best_x
