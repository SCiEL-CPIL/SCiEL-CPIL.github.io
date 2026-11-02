"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26
DEADLINE = 115.0  # seconds; keep safety margin before 120s limit


def _make_constraints(n):
    """Build constraint functions for non-overlap and wall constraints."""
    # Number of pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def constraint_array(x):
        # x is flattened: [x0,y0,r0, x1,y1,r1, ...]
        # Constraints:
        # 1. For each pair (i,j): (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        # 2. For each circle i: ri - xi >= 0  (left wall)
        # 3. For each circle i: 1 - ri - xi >= 0  (right wall)
        # 4. For each circle i: ri - yi >= 0  (bottom wall)
        # 5. For each circle i: 1 - ri - yi >= 0  (top wall)
        # 6. ri >= 0 (radius non-negative) -- handled by bounds

        cx = x[0::3]
        cy = x[1::3]
        cr = x[2::3]

        c = np.empty(len(pairs) + 5 * n)
        idx = 0
        for (i, j) in pairs:
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            c[idx] = dx * dx + dy * dy - (cr[i] + cr[j]) ** 2
            idx += 1
        for i in range(n):
            c[idx] = cr[i] - cx[i]
            idx += 1
            c[idx] = 1.0 - cr[i] - cx[i]
            idx += 1
            c[idx] = cr[i] - cy[i]
            idx += 1
            c[idx] = 1.0 - cr[i] - cy[i]
            idx += 1
        return c

    return constraint_array


def _random_initial(n, rng):
    """Generate a random feasible-ish initial configuration."""
    # Start with small random circles
    x = np.zeros(3 * n)
    for i in range(n):
        x[3*i] = rng.uniform(0.1, 0.9)
        x[3*i+1] = rng.uniform(0.1, 0.9)
        x[3*i+2] = rng.uniform(0.02, 0.08)
    return x


def _sum_radii(x):
    return -np.sum(x[2::3])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    n = N
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n  # x, y, r bounds

    constraint_func = _make_constraints(n)

    best_x = None
    best_val = -np.inf

    # Multiple restarts with different initial conditions
    n_restarts = 30
    max_time_per_restart = (DEADLINE - (time.time() - start_time)) / n_restarts - 2.0

    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break

        # Create diverse initial conditions
        if restart == 0:
            # Grid-based initial
            side = int(np.ceil(np.sqrt(n)))
            x0 = np.zeros(3 * n)
            for i in range(n):
                row = i // side
                col = i % side
                x0[3*i] = (col + 0.5) / side
                x0[3*i+1] = (row + 0.5) / side
                x0[3*i+2] = 0.5 / side - 0.01
        elif restart == 1:
            # Random
            x0 = _random_initial(n, rng)
        else:
            # Perturb best so far or random
            if best_x is not None and rng.random() < 0.5:
                x0 = best_x + rng.normal(0, 0.02, 3*n)
                x0 = np.clip(x0, 0, 1)
                x0[2::3] = np.clip(x0[2::3], 0.01, 0.5)
            else:
                x0 = _random_initial(n, rng)

        # Ensure initial feasibility approximately: shrink radii if needed
        cx = x0[0::3]
        cy = x0[1::3]
        cr = x0[2::3]
        # Check wall constraints
        cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
        cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
        cr = np.maximum(cr, 0.005)
        x0[2::3] = cr

        # Run SLSQP
        try:
            res = minimize(
                _sum_radii,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': constraint_func}],
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            if res.success or res.fun < best_val:
                # Verify feasibility
                c = constraint_func(res.x)
                if np.all(c >= -1e-6):
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
        except Exception:
            pass

        # Check time
        if time.time() - start_time > DEADLINE - 15:
            break

    if best_x is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:n])

    # Final check and cleanup
    cx = best_x[0::3]
    cy = best_x[1::3]
    cr = best_x[2::3]

    # Ensure strict feasibility
    for i in range(n):
        cr[i] = min(cr[i], cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        cr[i] = max(cr[i], 1e-6)
    best_x[2::3] = cr

    return best_x.reshape(n, 3)
