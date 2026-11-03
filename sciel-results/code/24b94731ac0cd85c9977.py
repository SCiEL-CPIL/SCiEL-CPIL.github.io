"""Circle packing in perimeter-4 rectangle: optimize positions, radii, and aspect ratio."""
import numpy as np
import time

N = 21
_DEADLINE = time.time() + 140.0  # safety margin under 150s


def _eval(x, aspect):
    """Return (score, valid). x = [cx0.., cy0.., r0..]."""
    cx = x[:N]
    cy = x[N:2 * N]
    r = x[2 * N:]
    W = 2.0 * aspect
    H = 2.0 / aspect
    # boundary feasibility
    if np.any(r < 0) or np.any(cx - r < -1e-9) or np.any(cx + r > W + 1e-9) \
       or np.any(cy - r < -1e-9) or np.any(cy + r > H + 1e-9):
        return 0.0, False
    # pairwise non-overlap
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    d = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    np.fill_diagonal(d, np.inf)
    np.fill_diagonal(need, 0.0)
    if np.any(d < need - 1e-9):
        return 0.0, False
    return float(np.sum(r)), True


def _barrier(x, aspect, mu):
    """Objective to MINIMIZE: -sum(r) - mu * log(feasibility_margin)."""
    cx = x[:N]
    cy = x[N:2 * N]
    r = x[2 * N:]
    W = 2.0 * aspect
    H = 2.0 / aspect
    # boundary margins (positive when feasible)
    bm = np.concatenate([
        cx - r, r - cx + W, cy - r, r - cy + H
    ])
    # pairwise overlap margins
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    pm = d - need
    np.fill_diagonal(pm, np.inf)
    # combine all margins into a single log-barrier
    margins = np.concatenate([bm, pm.ravel()])
    # clip to avoid log of non-positive
    margins = np.clip(margins, 1e-12, None)
    return -np.sum(r) - mu * np.sum(np.log(margins))


def _grid_start(aspect):
    """Build a feasible grid initial guess for given aspect."""
    W = 2.0 * aspect
    H = 2.0 / aspect
    cols = max(1, int(round(W / H * np.sqrt(N))))
    cols = max(1, min(cols, N))
    rows = int(np.ceil(N / cols))
    cw = W / cols
    ch = H / rows
    cells = []
    for i in range(cols):
        for j in range(rows):
            if len(cells) >= N:
                break
            cells.append((cw * (i + 0.5), ch * (j + 0.5), 0.45 * min(cw, ch)))
    cells = cells[:N]
    x = np.zeros(3 * N)
    for k, (px, py, rr) in enumerate(cells):
        x[k] = px
        x[N + k] = py
        x[2 * N + k] = rr
    return x


def construct():
    best = None
    best_score = -1.0

    # Baseline: seed grid (square)
    base = _grid_start(1.0)
    s, ok = _eval(base, 1.0)
    if ok:
        best = base.copy()
        best_score = s

    # Try multiple aspect ratios with SLSQP optimization
    aspects = [1.0, 0.85, 1.18, 1.4, 0.72, 1.6, 1.25, 0.6, 1.75]
    for aspect in aspects:
        if time.time() > _DEADLINE - 8:
            break
        x0 = _grid_start(aspect)
        if not _eval(x0, aspect)[1]:
            continue
        # variable bounds
        W = 2.0 * aspect
        H = 2.0 / aspect
        lb = np.zeros(3 * N)
        ub = np.zeros(3 * N)
        for k in range(N):
            lb[k] = 0.0
            ub[k] = W
            lb[N + k] = 0.0
            ub[N + k] = H
            lb[2 * N + k] = 1e-4
            ub[2 * N + k] = 0.5 * min(W, H)
        # anneal the barrier mu
        xcur = x0.copy()
        s, ok = _eval(xcur, aspect)
        if ok and s > best_score:
            best = xcur.copy()
            best_score = s
        for mu in [0.05, 0.01, 0.002, 0.0005]:
            if time.time() > _DEADLINE - 5:
                break
            try:
                from scipy.optimize import minimize
                res = minimize(_barrier, xcur, args=(aspect, mu), method='SLSQP',
                               bounds=list(zip(lb, ub)),
                               options={'maxiter': 300, 'ftol': 1e-9})
                xnew = res.x
                snew, ok = _eval(xnew, aspect)
                if ok:
                    xcur = xnew
                    if snew > best_score:
                        best = xnew.copy()
                        best_score = snew
                else:
                    # fall back to last feasible
                    break
            except Exception:
                break
        # local polish: try to grow radii via small SLSQP on sum(r) with constraints
        if time.time() > _DEADLINE - 5:
            break
        try:
            from scipy.optimize import minimize
            cons = []
            W = 2.0 * aspect
            H = 2.0 / aspect
            cx = xcur[:N]; cy = xcur[N:2*N]; r = xcur[2*N:]
            # boundary constraints
            for k in range(N):
                cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[k] - x[2*N+k]})
                cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[2*N+k] - x[k] + W})
                cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[N+k] - x[2*N+k]})
                cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[2*N+k] - x[N+k] + H})
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                                 np.hypot(x[i]-x[j], x[N+i]-x[N+j]) - x[2*N+i] - x[2*N+j]})
            res = minimize(lambda x: -np.sum(x[2*N:]), xcur, method='SLSQP',
                           bounds=list(zip(lb, ub)), constraints=cons,
                           options={'maxiter': 400, 'ftol': 1e-10})
            xnew = res.x
            snew, ok = _eval(xnew, aspect)
            if ok and snew > best_score:
                best = xnew.copy()
                best_score = snew
        except Exception:
            pass

    if best is None:
        # absolute fallback: seed
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # build (x, y, r) output
    out = np.zeros((N, 3))
    out[:, 0] = best[:N]
    out[:, 1] = best[N:2 * N]
    out[:, 2] = best[2 * N:]
    return out
