"""Circle packing in unit square - 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 150.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _objective(x):
    """Negative sum of radii (we minimize)."""
    # x = [x0,y0,r0, x1,y1,r1, ...]
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    return -np.sum(rs)


def _constraints_func(x):
    """Return list of constraint functions. Each constraint >= 0."""
    constraints = []
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]

    # Boundary constraints: circles must be inside [0,1]^2
    for i in range(N):
        constraints.append(lambda x, i=i: x[3*i] - x[3*i+2])  # x - r >= 0
        constraints.append(lambda x, i=i: x[3*i+1] - x[3*i+2])  # y - r >= 0
        constraints.append(lambda x, i=i: 1.0 - x[3*i] - x[3*i+2])  # 1-x-r >= 0
        constraints.append(lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2])  # 1-y-r >= 0

    # Non-overlap constraints: dist(i,j) >= ri + rj
    for i in range(N):
        for j in range(i+1, N):
            constraints.append(lambda x, i=i, j=j:
                (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2)

    return constraints


def _make_constraint_list():
    """Build scipy constraint dicts once."""
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]})
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2})
    return cons


def _corner_heavy_init(seed):
    """Initialize with larger circles in corners, smaller in center."""
    rng = np.random.RandomState(seed)
    x = np.zeros(3 * N)

    # Corner positions (4 corners)
    corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]
    # Edge midpoints
    edges = [(0.5, 0.06), (0.5, 0.94), (0.06, 0.5), (0.94, 0.5)]
    # Remaining positions scattered
    remaining = 26 - 4 - 4

    idx = 0
    # Corners: larger radii
    for (cx, cy) in corners:
        r = 0.15 + rng.random() * 0.05
        x[3*idx] = cx + rng.random() * 0.02 - 0.01
        x[3*idx+1] = cy + rng.random() * 0.02 - 0.01
        x[3*idx+2] = r
        idx += 1

    # Edges: medium radii
    for (ex, ey) in edges:
        r = 0.10 + rng.random() * 0.05
        x[3*idx] = ex + rng.random() * 0.02 - 0.01
        x[3*idx+1] = ey + rng.random() * 0.02 - 0.01
        x[3*idx+2] = r
        idx += 1

    # Remaining: smaller, scattered
    for _ in range(remaining):
        px = 0.1 + 0.8 * rng.random()
        py = 0.1 + 0.8 * rng.random()
        r = 0.06 + rng.random() * 0.06
        x[3*idx] = px
        x[3*idx+1] = py
        x[3*idx+2] = r
        idx += 1

    return x


def _grid_init(seed):
    """Grid-based initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(3 * N)
    for i in range(N):
        row = i // side
        col = i % side
        x[3*i] = (col + 0.5) / side + rng.random() * 0.02 - 0.01
        x[3*i+1] = (row + 0.5) / side + rng.random() * 0.02 - 0.01
        x[3*i+2] = 0.5 / side - 0.01
    return x


def _random_init(seed):
    """Fully random initialization."""
    rng = np.random.RandomState(seed)
    x = np.zeros(3 * N)
    for i in range(N):
        x[3*i] = 0.1 + 0.8 * rng.random()
        x[3*i+1] = 0.1 + 0.8 * rng.random()
        x[3*i+2] = 0.05 + 0.1 * rng.random()
    return x


def _is_valid(x, tol=1e-6):
    """Check if configuration is valid (no overlaps, all inside square)."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]

    # Boundary check
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False

    # Overlap check
    for i in range(N):
        for j in range(i+1, N):
            d2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
            if d2 < (rs[i]+rs[j])**2 - tol:
                return False
    return True


def _fix_boundaries(x):
    """Clamp radii and positions to stay within [0,1]^2."""
    x = x.copy()
    for i in range(N):
        r = x[3*i+2]
        # Clamp radius
        r = max(0.001, r)
        # Clamp position
        x[3*i] = np.clip(x[3*i], r, 1.0 - r)
        x[3*i+1] = np.clip(x[3*i+1], r, 1.0 - r)
        x[3*i+2] = r
    return x


def _polish(x, time_limit=10.0):
    """Polish a feasible or near-feasible solution with SLSQP."""
    x = _fix_boundaries(x)
    cons = _make_constraint_list()

    def obj(x):
        return -np.sum(x[2::3])

    def grad_obj(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    try:
        res = minimize(obj, x, method='SLSQP', jac=grad_obj,
                       bounds=[(0.001, 0.5)] * N,
                       constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})
        if res.success or res.fun < obj(x):
            x = _fix_boundaries(res.x)
    except Exception:
        pass

    # Second polish with tighter tolerance
    try:
        res2 = minimize(obj, x, method='SLSQP', jac=grad_obj,
                        bounds=[(0.001, 0.5)] * N,
                        constraints=cons,
                        options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
        if res2.success or res2.fun < obj(x):
            x = _fix_boundaries(res2.x)
    except Exception:
        pass

    return x


def _penalty_objective(x, mu=1.0):
    """Penalty-based objective for when constraints are too tight."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]

    # Base objective
    val = -np.sum(rs)

    # Boundary penalty
    for i in range(N):
        val += mu * max(0, rs[i] - xs[i])**2
        val += mu * max(0, rs[i] - ys[i])**2
        val += mu * max(0, rs[i] - (1.0 - xs[i]))**2
        val += mu * max(0, rs[i] - (1.0 - ys[i]))**2

    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            d2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
            overlap = (rs[i]+rs[j])**2 - d2
            if overlap > 0:
                val += mu * overlap**2

    return val


def _penalty_optimize(x, time_limit=8.0):
    """Optimize using penalty method to handle infeasible starts."""
    x = _fix_boundaries(x)
    mu = 1.0

    def obj_pen(x):
        return _penalty_objective(x, mu)

    def grad_pen(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            x_minus = x.copy()
            x_minus[k] -= eps
            g[k] = (obj_pen(x_plus) - obj_pen(x_minus)) / (2*eps)
        return g

    best_x = x.copy()
    best_val = obj_pen(x)

    for _ in range(5):
        if _remaining() < time_limit:
            break
        try:
            res = minimize(obj_pen, x, method='L-BFGS-B', jac=grad_pen,
                           bounds=[(0.001, 0.5)] * N,
                           options={'maxiter': 100, 'ftol': 1e-8, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
            x = res.x
        except Exception:
            break

    return _fix_boundaries(best_x)


def construct():
    global START_TIME
    START_TIME = time.time()

    best_x = None
    best_score = -np.inf

    # Try multiple initializations
    inits = []

    # Corner-heavy initializations
    for seed in [42, 123, 456, 789, 1000]:
        inits.append(_corner_heavy_init(seed))

    # Grid initialization
    inits.append(_grid_init(42))

    # Random initializations
    for seed in [7, 13, 99]:
        inits.append(_random_init(seed))

    cons = _make_constraint_list()

    for init_x in inits:
        if _remaining() < 12.0:
            break

        init_x = _fix_boundaries(init_x)

        # First try penalty optimization to get close to feasibility
        x_pen = _penalty_optimize(init_x, time_limit=min(6.0, _remaining() - 2.0))

        # Then try SLSQP polish from the penalty result
        x_polished = _polish(x_pen, time_limit=min(8.0, _remaining() - 2.0))

        # Also try direct SLSQP from original init
        if _remaining() > 10.0:
            try:
                res = minimize(lambda x: -np.sum(x[2::3]), init_x,
                               method='SLSQP',
                               jac=lambda x: np.concatenate([np.zeros(2*N), np.ones(N)] * 1) if False else None,
                               bounds=[(0.001, 0.5)] * N,
                               constraints=cons,
                               options={'maxiter': 150, 'ftol': 1e-10, 'disp': False})
                if res.fun < _objective(x_polished):
                    x_polished = _fix_boundaries(res.x)
            except Exception:
                pass

        # Evaluate
        if _is_valid(x_polished, tol=1e-5):
            score = -_objective(x_polished)
            if score > best_score:
                best_score = score
                best_x = x_polished.copy()

        # Also check penalty result
        if _is_valid(x_pen, tol=1e-5):
            score = -_objective(x_pen)
            if score > best_score:
                best_score = score
                best_x = x_pen.copy()

    # Final high-precision polish on best solution
    if best_x is not None and _remaining() > 8.0:
        best_x = _polish(best_x, time_limit=min(10.0, _remaining() - 2.0))

    # If we still don't have a valid solution, use a safe fallback
    if best_x is None or not _is_valid(best_x, tol=1e-4):
        # Fallback: small non-overlapping circles
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_x = np.array(cells[:N]).flatten()

    # Ensure validity with strict check
    if not _is_valid(best_x, tol=1e-4):
        # Slightly shrink all radii to ensure validity
        best_x = best_x.copy()
        for i in range(N):
            best_x[3*i+2] *= 0.99
            best_x[3*i] = np.clip(best_x[3*i], best_x[3*i+2], 1.0 - best_x[3*i+2])
            best_x[3*i+1] = np.clip(best_x[3*i+1], best_x[3*i+2], 1.0 - best_x[3*i+2])

    # Return as (N, 3) array
    result = best_x.reshape(N, 3)
    return result
