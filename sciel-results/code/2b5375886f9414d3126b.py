"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: kill after 150s wall-clock, return best found with safety margin
_DEADLINE = 150.0
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def _validity(arr):
    """Check if circles are disjoint and inside [0,1]^2."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2 < (ri + rj) ** 2 - 1e-7:
                return False
    return True


def _objective(x):
    """Negative sum of radii (minimize)."""
    # x is flat array of length 3*n: (x0,y0,r0, x1,y1,r1, ...)
    return -np.sum(x[2::3])


def _constraints(n):
    """Build constraint functions for SLSQP: non-overlap and wall containment."""
    cons = []
    # Wall constraints: for each circle i
    for i in range(n):
        xi, yi, ri = 3 * i, 3 * i + 1, 3 * i + 2
        # x_i >= r_i  =>  x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, xi=xi, ri=ri: x[xi] - x[ri]})
        # x_i <= 1 - r_i  =>  1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, xi=xi, ri=ri: 1.0 - x[xi] - x[ri]})
        # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, yi=yi, ri=ri: x[yi] - x[ri]})
        # y_i <= 1 - r_i
        cons.append({'type': 'ineq', 'fun': lambda x, yi=yi, ri=ri: 1.0 - x[yi] - x[ri]})
        # r_i >= 0.001 (small positive)
        cons.append({'type': 'ineq', 'fun': lambda x, ri=ri: x[ri] - 0.001})
    # Non-overlap: for each pair (i,j), (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # Equivalent: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = 3 * i, 3 * i + 1, 3 * i + 2
            xj, yj, rj = 3 * j, 3 * j + 1, 3 * j + 2
            cons.append({
                'type': 'ineq',
                'fun': lambda x, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj:
                    (x[xi] - x[xj]) ** 2 + (x[yi] - x[yj]) ** 2 - (x[ri] + x[rj]) ** 2
            })
    return cons


def _grid_seed(n, rng):
    """Generate a random-ish grid seed for n circles in unit square."""
    # Use a jittered grid to get a reasonable starting point
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5 + rng.uniform(-0.1, 0.1)) / side
            cy = (j + 0.5 + rng.uniform(-0.1, 0.1)) / side
            r = 0.45 / side
            cells.append((cx, cy, r))
    # If we don't have enough, add more
    while len(cells) < n:
        cells.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.02))
    arr = np.array(cells[:n])
    # Flatten
    return arr.flatten()


def _random_seed(n, rng):
    """Generate a random seed with small circles."""
    x = np.zeros(3 * n)
    for i in range(n):
        x[3 * i] = rng.uniform(0.1, 0.9)
        x[3 * i + 1] = rng.uniform(0.1, 0.9)
        x[3 * i + 2] = rng.uniform(0.02, 0.05)
    return x


def construct():
    n = N
    rng = np.random.default_rng(42)
    best_x = None
    best_val = -np.inf

    # Pre-compute constraints (they're static)
    constraints = _constraints(n)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n

    # Try multiple initializations
    n_starts = 12
    for start_idx in range(n_starts):
        if _check_deadline():
            break

        if start_idx == 0:
            x0 = _grid_seed(n, rng)
        else:
            x0 = _random_seed(n, rng)

        # Ensure initial feasibility (nudge if needed)
        # Simple projection: clip positions and reduce radii if overlapping
        for it in range(50):
            feasible = True
            for i in range(n):
                xi, yi, ri = x0[3*i], x0[3*i+1], x0[3*i+2]
                # Wall containment
                if xi < ri:
                    x0[3*i] = ri
                    feasible = False
                elif xi > 1 - ri:
                    x0[3*i] = 1 - ri
                    feasible = False
                if yi < ri:
                    x0[3*i+1] = ri
                    feasible = False
                elif yi > 1 - ri:
                    x0[3*i+1] = 1 - ri
                    feasible = False
                if ri < 0.001:
                    x0[3*i+2] = 0.001
                    feasible = False
            # Check overlaps
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi, ri = x0[3*i], x0[3*i+1], x0[3*i+2]
                    xj, yj, rj = x0[3*j], x0[3*j+1], x0[3*j+2]
                    d2 = (xi-xj)**2 + (yi-yj)**2
                    req = (ri+rj)**2
                    if d2 < req - 1e-6:
                        feasible = False
                        # Push apart
                        d = np.sqrt(max(d2, 1e-12))
                        overlap = (ri + rj) - d
                        if overlap > 0:
                            push = overlap / 2 + 1e-4
                            dx = (xj - xi) / d if d > 1e-12 else 1.0
                            dy = (yj - yi) / d if d > 1e-12 else 0.0
                            x0[3*i] -= dx * push
                            x0[3*i+1] -= dy * push
                            x0[3*j] += dx * push
                            x0[3*j+1] += dy * push
            if feasible:
                break

        # Run SLSQP
        try:
            res = minimize(
                _objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
        except Exception:
            continue

        if res.success or (res.fun < best_val):
            x_sol = res.x
            # Final validity check
            arr = x_sol.reshape(n, 3)
            if _validity(arr):
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best_x = x_sol.copy()

    if best_x is not None:
        return best_x.reshape(n, 3)

    # Fallback: grid
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
    while len(cells) < n:
        cells.append((0.5, 0.5, 0.01))
    return np.array(cells[:n])
# EVOLVE-BLOCK-END
