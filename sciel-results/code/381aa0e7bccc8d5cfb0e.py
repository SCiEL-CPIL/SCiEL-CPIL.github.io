"""Circle packing in unit square - local search with radius growth and position adjustment."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _init_positions(n):
    """Random initial positions in (0,1)^2 with small radii."""
    rng = np.random.default_rng()
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.03)
    return x, y, r


def _objective_and_grad(p, n):
    """
    Objective: minimize negative sum of radii + penalty for violations.
    p = [x0..x_{n-1}, y0..y_{n-1}, r0..r_{n-1}]
    """
    x = p[:n]
    y = p[n:2*n]
    r = p[2*n:]

    # Sum of radii (we maximize, so minimize negative)
    obj = -np.sum(r)

    # Pairwise non-overlap: dist_ij >= r_i + r_j
    # Use soft penalty
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy) + 1e-12
            violation = (r[i] + r[j]) - d
            if violation > 0:
                obj += 10.0 * violation * violation

    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        for val in [x[i] - r[i], 1.0 - x[i] - r[i], y[i] - r[i], 1.0 - y[i] - r[i]]:
            if val < 0:
                obj += 10.0 * val * val

    # Compute gradient
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)

    # d(-sum r)/dr_i = -1
    gr[:] = -1.0

    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy) + 1e-12
            violation = (r[i] + r[j]) - d
            if violation > 0:
                # penalty = 10 * violation^2
                # d(penalty)/d(x_i) = 10*2*violation * (-dx/d)
                # d(penalty)/d(x_j) = 10*2*violation * (dx/d)
                # d(penalty)/d(r_i) = 10*2*violation * 1
                # d(penalty)/d(r_j) = 10*2*violation * 1
                coeff = 20.0 * violation
                gx[i] += -coeff * dx / d
                gx[j] += coeff * dx / d
                gy[i] += -coeff * dy / d
                gy[j] += coeff * dy / d
                gr[i] += coeff
                gr[j] += coeff

    for i in range(n):
        # boundary: x_i - r_i
        v = x[i] - r[i]
        if v < 0:
            c = 20.0 * v
            gx[i] += c
            gr[i] += -c
        # 1 - x_i - r_i
        v = 1.0 - x[i] - r[i]
        if v < 0:
            c = 20.0 * v
            gx[i] += -c
            gr[i] += -c
        # y_i - r_i
        v = y[i] - r[i]
        if v < 0:
            c = 20.0 * v
            gy[i] += c
            gr[i] += -c
        # 1 - y_i - r_i
        v = 1.0 - y[i] - r[i]
        if v < 0:
            c = 20.0 * v
            gy[i] += -c
            gr[i] += -c

    grad = np.concatenate([gx, gy, gr])
    return obj, grad


def _is_valid(x, y, r, tol=1e-6):
    """Check if all circles are inside [0,1]^2 and non-overlapping."""
    for i in range(N):
        if r[i] < -tol or x[i] - r[i] < -tol or 1.0 - x[i] - r[i] < -tol:
            return False
        if y[i] - r[i] < -tol or 1.0 - y[i] - r[i] < -tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if np.sqrt(dx*dx + dy*dy) < r[i] + r[j] - tol:
                return False
    return True


def _fix_and_clamp(x, y, r):
    """Clamp positions and radii to be valid (shrink radii if needed)."""
    r = np.maximum(r, 1e-8)
    for i in range(N):
        # Ensure circle fits in [0,1]^2
        max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if r[i] > max_r:
            r[i] = max(0.0, max_r)
    return x, y, r


def construct():
    best_value = -np.inf
    best_result = None

    # Initial grid solution as fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    grid_arr = np.array(cells[:N])
    best_value = np.sum(grid_arr[:, 2])
    best_result = grid_arr

    rng = np.random.default_rng(42)

    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 5.0:
            break

        x, y, r = _init_positions(N)
        # Add some perturbation based on restart
        x += rng.normal(0, 0.1, N)
        y += rng.normal(0, 0.1, N)
        r = np.full(N, 0.02 + 0.01 * rng.random())

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)

        p0 = np.concatenate([x, y, r])

        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
        for i in range(N):
            bounds.append((0.0, 1.0))  # y
        for i in range(N):
            bounds.append((1e-6, 0.5))  # r

        try:
            res = minimize(
                _objective_and_grad,
                p0,
                args=(N,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue

        p = res.x
        x_opt = p[:N]
        y_opt = p[N:2*N]
        r_opt = p[2*N:]

        x_opt, y_opt, r_opt = _fix_and_clamp(x_opt, y_opt, r_opt)

        # Verify validity
        if _is_valid(x_opt, y_opt, r_opt):
            val = np.sum(r_opt)
            if val > best_value:
                best_value = val
                best_result = np.column_stack([x_opt, y_opt, r_opt])

        # Polish: try a few more iterations from the solution
        if time.time() < _DEADLINE - 3.0:
            try:
                res2 = minimize(
                    _objective_and_grad,
                    p,
                    args=(N,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                p2 = res2.x
                x2 = np.clip(p2[:N], 0.0, 1.0)
                y2 = np.clip(p2[N:2*N], 0.0, 1.0)
                r2 = np.maximum(p2[2*N:], 1e-6)
                x2, y2, r2 = _fix_and_clamp(x2, y2, r2)
                if _is_valid(x2, y2, r2):
                    val2 = np.sum(r2)
                    if val2 > best_value:
                        best_value = val2
                        best_result = np.column_stack([x2, y2, r2])
            except Exception:
                pass

    return best_result
