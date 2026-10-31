"""Circle packing in perimeter-4 rectangle: Aspect-ratio SLSQP + geometric pocket relocation."""
import numpy as np
import time
from scipy.optimize import minimize

try:
    from scipy.spatial import Delaunay
    _HAS_DEL = True
except Exception:
    _HAS_DEL = False

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_radius_at(cx, cy, r, W, H, i):
    """Max valid radius for circle i if others fixed."""
    r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
    r_max = r_wall
    for j in range(len(cx)):
        if i == j:
            continue
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        ra = d - r[j]
        if ra < r_max:
            r_max = ra
    return max(0.0, r_max)


def _max_radius_anywhere(cx, cy, r, W, H, i):
    """Find the best location (and radius) for circle i, ignoring its old position."""
    # Candidate points: current, center, corners-offset, and Voronoi vertices of others.
    others_x = np.delete(cx, i)
    others_y = np.delete(cy, i)
    others_r = np.delete(r, i)

    cands = [(cx[i], cy[i]), (W / 2.0, H / 2.0)]
    for mx in (0.0, W):
        for my in (0.0, H):
            cands.append((mx + 0.05, my + 0.05))
            cands.append((mx + W - 0.05, my + 0.05))
            cands.append((mx + 0.05, my + H - 0.05))
            cands.append((mx + W - 0.05, my + H - 0.05))

    if _HAS_DEL and len(others_x) >= 3:
        try:
            pts = np.column_stack((others_x, others_y))
            tri = Delaunay(pts)
            for simplex in tri.simplices:
                p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                # Circumcenter
                ax, ay = p0
                bx, by = p1
                cxv, cyv = p2
                d = 2.0 * (ax * (by - cyv) + bx * (cyv - ay) + cxv * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax * ax + ay * ay) * (by - cyv) + (bx * bx + by * by) * (cyv - ay) +
                      (cxv * cxv + cyv * cyv) * (ay - by)) / d
                uy = ((ax * ax + ay * ay) * (cxv - bx) + (bx * bx + by * by) * (ax - cxv) +
                      (cxv * cxv + cyv * cyv) * (bx - ax)) / d
                cands.append((ux, uy))
        except Exception:
            pass

    best_r = 0.0
    best_xy = (cx[i], cy[i])
    for (px, py) in cands:
        if px < 0 or px > W or py < 0 or py > H:
            continue
        r_wall = min(px, W - px, py, H - py)
        r_max = r_wall
        for k in range(len(others_x)):
            dx = px - others_x[k]
            dy = py - others_y[k]
            d = np.sqrt(dx * dx + dy * dy)
            ra = d - others_r[k]
            if ra < r_max:
                r_max = ra
        if r_max > best_r:
            best_r = r_max
            best_xy = (px, py)
    return best_xy, max(0.0, best_r)


def _aspect_slsqp(w0, h0, time_budget):
    """Solve for 21 equal circles in a w0 x h0 box; maximize common radius r."""
    def obj(v):
        x = v[0::2]
        y = v[1::2]
        r = v[-1]
        val = -r
        pen = 0.0
        # Wall
        pen += np.sum(np.maximum(r - x, 0.0) ** 2)
        pen += np.sum(np.maximum(x + r - w0, 0.0) ** 2)
        pen += np.sum(np.maximum(r - y, 0.0) ** 2)
        pen += np.sum(np.maximum(y + r - h0, 0.0) ** 2)
        # Pairwise
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        overlap = np.maximum(2.0 * r - np.sqrt(np.maximum(d2[iu], 0.0)), 0.0)
        pen += np.sum(overlap * overlap)
        return val + 1000.0 * pen

    # Grid init
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w0
        y[i] = (row + 0.5) / side * h0
    r0 = min(w0, h0) / side * 0.45
    v0 = np.zeros(2 * N + 1)
    v0[0::2] = x
    v0[1::2] = y
    v0[-1] = r0

    bounds = [(0, w0)] * N + [(0, h0)] * N + [(1e-4, 1.0)]
    # Interleave: x0,y0,x1,y1,...,r
    bounds2 = []
    for i in range(N):
        bounds2.append((0, w0))
        bounds2.append((0, h0))
    bounds2.append((1e-4, 1.0))

    maxiter = int(min(400, max(100, time_budget * 40)))
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds2,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        x = res.x[0::2]
        y = res.x[1::2]
        r = res.x[-1]
        if _is_valid(x, y, np.full(N, r), w0, h0):
            return x, y, np.full(N, r)
        # Repair
        r *= 0.98
        if _is_valid(x, y, np.full(N, r), w0, h0):
            return x, y, np.full(N, r)
    except Exception:
        pass
    return None


def _relocation_polish(cx, cy, r, W, H, time_budget, max_rounds=60):
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    for _ in range(max_rounds):
        if time.time() > time_budget:
            break
        # clearance of each circle = max radius - current radius
        clear = np.zeros(N)
        for i in range(N):
            clear[i] = _max_radius_at(cx, cy, r, W, H, i) - r[i]
        i = int(np.argmax(clear))
        if clear[i] < 1e-7:
            break
        new_xy, new_r = _max_radius_anywhere(cx, cy, r, W, H, i)
        if new_r > r[i] + 1e-7:
            cx[i] = new_xy[0]
            cy[i] = new_xy[1]
            r[i] = new_r
        else:
            break
    return cx, cy, r


def _uniform_grow(cx, cy, r, W, H, time_budget):
    for _ in range(200):
        if time.time() > time_budget:
            break
        r2 = r * 1.0005
        if _is_valid(cx, cy, r2, W, H):
            r = r2
        else:
            break
    return r


def construct():
    start = time.time()
    deadline = start + 140.0
    best = None
    best_val = 0.0

    # Phase 1: search aspect ratios with equal-circle SLSQP
    w_grid = np.arange(0.55, 1.46, 0.05)
    for w0 in w_grid:
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        rem = deadline - (time.time() - start) - 30
        if rem < 2:
            break
        sol = _aspect_slsqp(w0, h0, rem)
        if sol is None:
            continue
        cx, cy, r = sol
        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best = (cx.copy(), cy.copy(), r.copy(), w0, h0)

    if best is None:
        # Fallback grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cells.append(((col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    cx, cy, r, W, H = best

    # Phase 2: geometric pocket relocation (the structural core)
    if time.time() < deadline - 15:
        cx, cy, r = _relocation_polish(cx, cy, r, W, H, deadline - (time.time() - start) - 12)
        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best = (cx.copy(), cy.copy(), r.copy(), W, H)

    # Phase 3: uniform grow
    if time.time() < deadline - 4:
        cx, cy, r, W, H = best
        r = _uniform_grow(cx, cy, r, W, H, deadline - (time.time() - start))
        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best = (cx.copy(), cy.copy(), r.copy(), W, H)

    # Phase 4: one more relocation pass with leftover time
    if time.time() < deadline - 2:
        cx, cy, r, W, H = best
        cx, cy, r = _relocation_polish(cx, cy, r, W, H, deadline - (time.time() - start), max_rounds=20)
        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best = (cx.copy(), cy.copy(), r.copy(), W, H)

    cx, cy, r, W, H = best

    # Final safety shrink
    for _ in range(20):
        r2 = r * (1 - 1e-9)
        if _is_valid(cx, cy, r2, W, H):
            r = r2
        else:
            break

    return np.column_stack((cx, cy, r))
