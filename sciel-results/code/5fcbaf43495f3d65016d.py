"""Circle packing in unit square, n=32. Gradient-based optimizer."""
import time
import numpy as np

N = 32
_T0 = time.time()
_DEADLINE = _T0 + 140.0  # safety margin under the 150 s kill


def _feasible(arr):
    """Check circles are inside [0,1]^2 with disjoint interiors."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r < 0) or np.any(r > 1):
        return False
    if np.any(x < -1e-9) or np.any(y < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9):
        return False
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-9:
                return False
    return True


def _score(arr):
    return float(np.sum(arr[:, 2]))


def _hex_init(n):
    """Hexagonal (close-packed) grid of equal circles, first n used."""
    R = 0.05
    dx = 2 * R
    dy = R * np.sqrt(3.0)
    pts = []
    row = 0
    while len(pts) < n:
        off = (row % 2) * R
        col = 0
        while len(pts) < n:
            x = R + off + col * dx
            y = R + row * dy
            if x <= 1 - R + 1e-9 and y <= 1 - R + 1e-9:
                pts.append((x, y))
            col += 1
        row += 1
    pts = pts[:n]
    arr = np.array([(p[0], p[1], R) for p in pts], dtype=float)
    return arr


def _gradient(arr, beta):
    """Return (dsum, dF) where F = sum of soft-constraint violations, dF = dF/d(arr)."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    n = len(arr)

    dsum = np.zeros((n, 3))
    dF = np.zeros((n, 3))

    # boundary: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n):
        for c in range(4):
            if c == 0:
                g = x[i] - r[i]
                gx, gy, gr = 1.0, 0.0, -1.0
            elif c == 1:
                g = 1.0 - x[i] - r[i]
                gx, gy, gr = -1.0, 0.0, -1.0
            elif c == 2:
                g = y[i] - r[i]
                gx, gy, gr = 0.0, 1.0, -1.0
            else:
                g = 1.0 - y[i] - r[i]
                gx, gy, gr = 0.0, -1.0, -1.0
            if g < 0:
                w = -beta * np.exp(beta * g)
                dF[i, 0] += w * gx
                dF[i, 1] += w * gy
                dF[i, 2] += w * gr

    # pair overlap: dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            g = d - r[i] - r[j]
            if g < 0:
                w = -beta * np.exp(beta * g)
                if d > 1e-12:
                    ux, uy = dx / d, dy / d
                else:
                    ux, uy = 1.0, 0.0
                dF[i, 0] += w * ux
                dF[i, 1] += w * uy
                dF[i, 2] += w * 1.0
                dF[j, 0] += w * (-ux)
                dF[j, 1] += w * (-uy)
                dF[j, 2] += w * 1.0

    return dsum, dF


def _project(arr):
    """Cheap projection: keep circles inside the square (shrink r, clamp centers)."""
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    for it in range(6):
        r = np.clip(r, 0.0, 1.0)
        for i in range(len(arr)):
            r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            r[i] = max(r[i], 0.0)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
    arr[:, 0] = x
    arr[:, 1] = y
    arr[:, 2] = r
    return arr


def _optimize(arr, beta):
    """Gradient ascent on  sum(r) - F, with line search and projection."""
    n = len(arr)
    best = arr.copy()
    best_val = _score(best)
    F0 = 0.0
    step = 0.02
    for it in range(4000):
        if time.time() > _DEADLINE:
            break
        dsum, dF = _gradient(arr, beta)
        grad = dsum - dF
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        # line search
        cand = arr.copy()
        found = False
        s = step
        for _ in range(12):
            cand = arr + s * grad
            cand = _project(cand)
            if _feasible(cand):
                v = _score(cand)
                if v > best_val + 1e-12:
                    arr = cand
                    best = cand.copy()
                    best_val = v
                found = True
                break
            s *= 0.5
        if not found:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            step = min(step * 1.15, 0.05)
    return best


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140.0

    best = _hex_init(N)
    best_val = _score(best)

    # warm-up: increase beta to tighten the soft constraints
    for beta in [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]:
        if time.time() > _DEADLINE:
            break
        cur = _optimize(best.copy(), beta)
        if _score(cur) > best_val:
            best = cur
            best_val = _score(best)

    # final high-precision polish with tight constraints
    if time.time() < _DEADLINE:
        cur = _optimize(best.copy(), 200.0)
        if _score(cur) > best_val:
            best = cur
            best_val = _score(cur)

    # ensure a valid result; fall back to the trivial grid if needed
    if not _feasible(best):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N], dtype=float)

    return best
