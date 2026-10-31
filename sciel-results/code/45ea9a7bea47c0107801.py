"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time

N = 26
TIME_LIMIT = 150.0
SAFETY = 5.0          # seconds to keep before the hard kill
EPS = 1e-10


def _feasible(x, y, r):
    """Return True if circles are inside the unit square and pairwise disjoint."""
    if np.any(r < 0) or np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    # inside
    if np.any(r > x) or np.any(r > 1 - x) or np.any(r > y) or np.any(r > 1 - y):
        return False
    # pairwise
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d + 1e-9 < r[i] + r[j]:
                return False
    return True


def _score(x, y, r):
    """Sum of radii if feasible, else 0."""
    if _feasible(x, y, r):
        return float(np.sum(r))
    return 0.0


def _init(seed):
    """Deterministic random-ish initial placement (seeded)."""
    rng = np.random.default_rng(seed)
    # jittered grid to start near a reasonable packing
    side = int(np.ceil(np.sqrt(N)))
    xs = np.empty(N)
    ys = np.empty(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            xs[idx] = (i + 0.5) / side + rng.uniform(-0.2, 0.2) / side
            ys[idx] = (j + 0.5) / side + rng.uniform(-0.2, 0.2) / side
            idx += 1
    xs = np.clip(xs, 0.05, 0.95)
    ys = np.clip(ys, 0.05, 0.95)
    return xs, ys


def _penalty_optimize(x0, y0, r0, deadline, beta0=200.0, beta_growth=1.3):
    """Gradient ascent on positions minimizing the sum of contact/violation penalties.

    Objective (to maximize):  sum(r_i) - beta * (penalties)
    Penalties:
      - wall violation:  max(0, r_i - x_i)^2 + ... for each wall
      - pair overlap:    max(0, r_i + r_j - dist_ij)^2
    We use a smooth approximation via softplus for stability, but squared hinge
    works well. We use L-BFGS-style via scipy if available, else manual GD.
    """
    from scipy.optimize import minimize

    x = x0.copy()
    y = y0.copy()
    r = r0.copy()

    def neg_obj(vars_):
        X = vars_[:N]
        Y = vars_[N:2 * N]
        # wall terms
        wall = (np.maximum(0.0, r - X) ** 2 +
                np.maximum(0.0, r - (1 - X)) ** 2 +
                np.maximum(0.0, r - Y) ** 2 +
                np.maximum(0.0, r - (1 - Y)) ** 2)
        # pair terms
        pair = 0.0
        for i in range(N):
            dx = X[i] - X[i + 1:]
            dy = Y[i] - Y[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + EPS)
            viol = np.maximum(0.0, r[i] + r[i + 1:] - d)
            pair += np.sum(viol ** 2)
        return -(np.sum(r) - beta0 * (wall + pair))

    def neg_grad(vars_):
        X = vars_[:N].copy()
        Y = vars_[N:2 * N].copy()
        gx = np.zeros(N)
        gy = np.zeros(N)
        # wall gradients
        # d/dX of max(0, r - X)^2 = -2 max(0, r-X)
        gx += -2.0 * np.maximum(0.0, r - X)
        gx += 2.0 * np.maximum(0.0, r - (1 - X))
        gy += -2.0 * np.maximum(0.0, r - Y)
        gy += 2.0 * np.maximum(0.0, r - (1 - Y))
        # pair gradients
        for i in range(N):
            dx = X[i] - X[i + 1:]
            dy = Y[i] - Y[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + EPS)
            viol = np.maximum(0.0, r[i] + r[i + 1:] - d)
            # d/dX[i] of viol^2 = 2*viol * d/dX[i](r_i+r_j - d) = 2*viol * (dx/d)
            gx[i] += 2.0 * np.sum(viol * dx / d)
            gx[i + 1:] -= 2.0 * viol * dx / d
            gy[i] += 2.0 * np.sum(viol * dy / d)
            gy[i + 1:] -= 2.0 * viol * dy / d
        # scale by beta0
        gx = -beta0 * gx
        gy = -beta0 * gy
        return np.concatenate([gx, gy])

    best = None
    best_val = 0.0
    beta = beta0
    x_cur = x0.copy()
    y_cur = y0.copy()

    for _iter in range(40):
        if time.time() > deadline:
            break
        var0 = np.concatenate([x_cur, y_cur])
        try:
            res = minimize(neg_obj, var0, jac=neg_grad, method='L-BFGS-B',
                           options={'maxiter': 300, 'gtol': 1e-6, 'maxls': 50})
        except Exception:
            break
        x_cur = res.x[:N]
        y_cur = res.x[N:2 * N]
        x_cur = np.clip(x_cur, 0.0, 1.0)
        y_cur = np.clip(y_cur, 0.0, 1.0)
        val = _score(x_cur, y_cur, r)
        if val > best_val:
            best_val = val
            best = (x_cur.copy(), y_cur.copy())
        # grow penalty if still infeasible
        if not _feasible(x_cur, y_cur, r):
            beta = min(beta * beta_growth, 5e4)
        else:
            # feasible; keep going a bit to polish but don't over-penalize
            break

    if best is None:
        return x0, y0, r
    return best[0], best[1], r


def _lp_radius(x, y):
    """Given positions, find max sum of radii s.t. circles inside square and disjoint.

    This is a linear program in variables r_i >= 0:
      maximize   sum r_i
      s.t.       r_i <= x_i
                 r_i <= 1 - x_i
                 r_i <= y_i
                 r_i <= 1 - y_i
                 r_i + r_j <= d_ij   for all i<j
    """
    from scipy.optimize import linprog

    d = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d[i, j] = d[j, i] = np.hypot(x[i] - x[j], y[i] - y[j])

    c = -np.ones(N)  # minimize -sum r
    A_ub = []
    b_ub = []
    # wall constraints: r_i <= x_i  ->  -r_i >= -x_i  ->  r_i - x_i <= 0
    # linprog uses A_ub @ var <= b_ub
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row); b_ub.append(x[i])
        row = np.zeros(N); row[i] = 1.0
        A_ub.append(row); b_ub.append(1 - x[i])
        row = np.zeros(N); row[i] = 1.0
        A_ub.append(row); b_ub.append(y[i])
        row = np.zeros(N); row[i] = 1.0
        A_ub.append(row); b_ub.append(1 - y[i])
    # pair constraints: r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i + 1, N):
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d[i, j])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * N
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.clip(res.x, 0.0, None)
    except Exception:
        pass
    # fallback: heuristic radii
    r = np.empty(N)
    for i in range(N):
        r[i] = min(x[i], 1 - x[i], y[i], 1 - y[i])
        for j in range(N):
            if i != j:
                r[i] = min(r[i], (d[i, j] - r[j]) if r[j] < d[i, j] else r[i])
    return np.clip(r, 0.0, None)


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY

    best = None
    best_val = 0.0

    # Try several seeds with the penalty optimizer, then LP polish.
    seeds = [0, 1, 2, 3, 7, 11, 42, 123, 999, 2024, 55, 888, 1000, 31415, 27182]
    for s in seeds:
        if time.time() > deadline:
            break
        x0, y0 = _init(s)
        # initial radii: small, from wall limits
        r0 = np.minimum(np.minimum(x0, 1 - x0), np.minimum(y0, 1 - y0)) * 0.8
        x1, y1, r1 = _penalty_optimize(x0, y0, r0, deadline)
        # LP polish for radii given positions
        rp = _lp_radius(x1, y1)
        val = _score(x1, y1, rp)
        if val > best_val:
            best_val = val
            best = (x1.copy(), y1.copy(), rp.copy())

    # Final polish loop on the best: re-run LP then a light position re-opt with
    # the LP radii to unlock more room, then LP again.
    if best is not None and time.time() < deadline:
        x, y, r = best
        for _ in range(6):
            if time.time() > deadline:
                break
            # light position optimization using current radii
            x2, y2, _ = _penalty_optimize(x, y, r, deadline)
            r2 = _lp_radius(x2, y2)
            val = _score(x2, y2, r2)
            if val > best_val:
                best_val = val
                best = (x2.copy(), y2.copy(), r2.copy())
                x, y, r = x2, y2, r2
            else:
                break

    if best is None:
        # last-resort grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
        return arr

    x, y, r = best
    out = np.column_stack([x, y, r])
    # final safety check; if infeasible, shrink radii
    if not _feasible(x, y, r):
        # shrink to feasibility
        scale = 1.0
        for _ in range(50):
            rs = r * scale
            if _feasible(x, y, rs):
                r = rs
                break
            scale *= 0.95
        out = np.column_stack([x, y, r])
    return out
