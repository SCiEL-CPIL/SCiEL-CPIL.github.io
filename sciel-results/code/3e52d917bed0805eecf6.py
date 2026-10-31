"""Circle packing in perimeter-4 rectangle.
Maximize sum of radii of 21 disjoint circles.
Uses SLSQP with aspect-ratio parameterization and multi-start.
"""
import time
import numpy as np

N = 21
DEADLINE = 115.0  # safety margin under 120s limit
_t0 = time.time()


def _remaining():
    return DEADLINE - (time.time() - _t0)


def _sides(t):
    """Given aspect ratio t = a/b, return (a, b) with a+b=2."""
    d = np.sqrt(1.0 + t * t)
    return t / d, 1.0 / d


def _make_constraints(a, b):
    """Return list of inequality constraint dicts for SLSQP.
    Variables: x (N), y (N), r (N)  -> total 3N.
    Constraints (g >= 0):
      overlap: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
      left:   x_i - r_i >= 0
      right:  a - x_i - r_i >= 0
      bottom: y_i - r_i >= 0
      top:    b - y_i - r_i >= 0
    """
    cons = []
    idx = np.arange(N)
    for i in range(N):
        for j in range(i + 1, N):
            def overlap(zz, i=i, j=j):
                dx = zz[i] - zz[N + j]
                dy = zz[N + i] - zz[N + N + j]
                rr = zz[2 * N + i] + zz[2 * N + j]
                return dx * dx + dy * dy - rr * rr
            cons.append({'type': 'ineq', 'fun': overlap})
    # boundaries
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda zz, i=i: zz[i] - zz[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda zz, i=i: a - zz[i] - zz[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda zz, i=i: zz[N + i] - zz[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda zz, i=i: b - zz[N + i] - zz[2 * N + i]})
    return cons


def _initial_layout(a, b, seed):
    rng = np.random.default_rng(seed)
    # Grid-based initial layout with jitter
    cols = 5
    rows = 5
    xs = []
    ys = []
    rs = []
    for i in range(cols):
        for j in range(rows):
            if len(xs) >= N:
                break
            cx = (i + 0.5) * a / cols
            cy = (j + 0.5) * b / rows
            r0 = 0.5 * min(a / cols, b / rows)
            xs.append(cx + rng.uniform(-0.02, 0.02) * a)
            ys.append(cy + rng.uniform(-0.02, 0.02) * b)
            rs.append(r0 * 0.9)
    # pad if needed
    while len(xs) < N:
        xs.append(a * 0.5)
        ys.append(b * 0.5)
        rs.append(0.05)
    return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])


def _is_feasible(zz, a, b, tol=1e-4):
    x, y, r = zz[:N], zz[N:2*N], zz[2*N:3*N]
    if np.any(r < -tol):
        return False
    if np.any(x < -tol) or np.any(x > a + tol) or np.any(y < -tol) or np.any(y > b + tol):
        return False
    if np.any(r > x + tol) or np.any(r > a - x + tol) or np.any(r > y + tol) or np.any(r > b - y + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx * dx + dy * dy
            rr = r[i] + r[j]
            if dist2 < rr * rr - tol:
                return False
    return True


def _repair(zz, a, b):
    """Nudge radii down slightly and re-project to ensure feasibility."""
    x, y, r = zz[:N].copy(), zz[N:2*N].copy(), zz[2*N:3*N].copy()
    # clamp positions to [r, a-r] etc.
    for i in range(N):
        r[i] = max(0.0, r[i])
        x[i] = np.clip(x[i], r[i], a - r[i])
        y[i] = np.clip(y[i], r[i], b - r[i])
    # iteratively shrink overlapping circles
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                rr = r[i] + r[j]
                if dist < rr:
                    # shrink the larger one
                    if r[i] >= r[j]:
                        r[i] = max(0.0, r[i] - (rr - dist) / 2.0 * 0.9)
                    else:
                        r[j] = max(0.0, r[j] - (rr - dist) / 2.0 * 0.9)
                    changed = True
            # re-clamp position
            x[i] = np.clip(x[i], r[i], a - r[i])
            y[i] = np.clip(y[i], r[i], b - r[i])
        if not changed:
            break
    zz2 = np.concatenate([x, y, r])
    return zz2


def construct():
    global _t0
    _t0 = time.time()

    best_zz = None
    best_val = 0.0
    best_a, best_b = 1.0, 1.0

    # Try several aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 1.3, 0.7, 1.1, 1.4]
    seeds = [0, 1, 2, 3]

    for t in aspect_ratios:
        if _remaining() < 10.0:
            break
        a, b = _sides(t)
        for seed in seeds:
            if _remaining() < 8.0:
                break
            x0, y0, r0 = _initial_layout(a, b, seed)
            zz0 = np.concatenate([x0, y0, r0])
            cons = _make_constraints(a, b)
            bounds = [(0, a)] * N + [(0, b)] * N + [(0, 0.5)] * N

            def objective(zz):
                return -np.sum(zz[2 * N:3 * N])

            try:
                res = _slsqp_solve(objective, zz0, bounds, cons)
            except Exception:
                continue

            if res is None:
                continue
            zz = res
            # repair
            zz = _repair(zz, a, b)
            if not _is_feasible(zz, a, b):
                continue
            val = np.sum(zz[2 * N:3 * N])
            if val > best_val:
                best_val = val
                best_zz = zz.copy()
                best_a, best_b = a, b

    if best_zz is None:
        # Fallback: return the trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final repair and validation
    best_zz = _repair(best_zz, best_a, best_b)
    x, y, r = best_zz[:N], best_zz[N:2*N], best_zz[2*N:3*N]
    # Ensure strict feasibility with small epsilon
    eps = 1e-6
    for i in range(N):
        r[i] = max(0.0, r[i] - eps)
        x[i] = np.clip(x[i], r[i] + eps, best_a - r[i] - eps)
        y[i] = np.clip(y[i], r[i] + eps, best_b - r[i] - eps)
    # Final overlap check, shrink if needed
    for _ in range(20):
        ok = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist2 = dx * dx + dy * dy
                rr = r[i] + r[j] + eps
                if dist2 < rr * rr:
                    # shrink both slightly
                    shrink = (rr - np.sqrt(dist2)) / 2.0 + eps
                    r[i] = max(0.0, r[i] - shrink)
                    r[j] = max(0.0, r[j] - shrink)
                    ok = False
        if ok:
            break

    result = np.column_stack([x, y, r])
    return result


def _slsqp_solve(objective, zz0, bounds, cons):
    from scipy.optimize import minimize
    res = minimize(objective, zz0, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 200, 'ftol': 1e-8})
    if _remaining() < 5.0:
        return None
    if res.success or res.status == 0:
        return res.x
    # Even if not fully successful, check if it improved
    return res.x


if __name__ == '__main__':
    res = construct()
    print("Result shape:", res.shape)
    print("Sum of radii:", np.sum(res[:, 2]))
    # Validate
    x, y, r = res[:, 0], res[:, 1], res[:, 2]
    print("Max x:", np.max(x), "Min x:", np.min(x))
    print("Max y:", np.max(y), "Min y:", np.min(y))
    # Check perimeter: we need to infer a,b from the layout
    # a = max(x) + max(r where x is max) ... actually a = max_i (x_i + r_i) if circle touches right
    # Simpler: a = max(x + r), b = max(y + r)
    a_est = np.max(x + r)
    b_est = np.max(y + r)
    print("Estimated sides:", a_est, b_est, "Perimeter:", 2 * (a_est + b_est))
