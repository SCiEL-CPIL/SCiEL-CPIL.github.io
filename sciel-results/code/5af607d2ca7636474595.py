import numpy as np
import time
import random

N = 21
START = time.time()
DEADLINE = START + 144.0

def _time_ok():
    return time.time() < DEADLINE

def _max_radius_at(px, py, cx, cy, r, W, H):
    """Largest radius for a circle centered at (px,py) not overlapping existing
       circles nor the rectangle boundary."""
    best = min(px, W - px, py, H - py)
    if best <= 0:
        return 0.0
    for i in range(len(r)):
        dx = px - cx[i]
        dy = py - cy[i]
        d = np.hypot(dx, dy)
        lim = d - r[i]
        if lim < best:
            best = lim
    if best <= 0:
        return 0.0
    return best

def _place_circle(W, H, cx, cy, r, rng, n_samples=400, n_refine=120):
    """Find the largest feasible circle in current free space and place it.
       Returns (px, py, rad) or None if no room."""
    # candidate centers: random + grid
    cand = []
    cand.append((W/2, H/2))
    # coarse grid
    for gy in range(1, 9):
        for gx in range(1, 9):
            cand.append((gx*W/9.0, gy*H/9.0))
    # random
    for _ in range(n_samples):
        cand.append((rng.uniform(0.05*W, 0.95*W), rng.uniform(0.05*H, 0.95*H)))

    best_rad = 0.0
    best_px = best_py = None
    # evaluate candidates
    for (px, py) in cand:
        rad = _max_radius_at(px, py, cx, cy, r, W, H)
        if rad > best_rad:
            best_rad = rad
            best_px, best_py = px, py

    if best_rad <= 1e-6:
        return None

    # refine around the best point: try to push center outward to increase radius
    # local search: nudge center to maximize radius
    for _ in range(n_refine):
        improved = False
        for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
            for step in (best_rad*0.25, best_rad*0.1, best_rad*0.04, 0.005):
                nxp = best_px + step*np.cos(ang)
                nyp = best_py + step*np.sin(ang)
                if nxp < 0 or nxp > W or nyp < 0 or nyp > H:
                    continue
                rad = _max_radius_at(nxp, nyp, cx, cy, r, W, H)
                if rad > best_rad + 1e-9:
                    best_rad = rad
                    best_px, best_py = nxp, nyp
                    improved = True
        if not improved:
            break

    if best_rad <= 1e-6:
        return None
    return (best_px, best_py, best_rad)

def _build_packing(W, H, order, rng, n_samples=300, n_refine=100):
    """Sequentially insert circles in given order at maximal radius."""
    cx = np.empty(N)
    cy = np.empty(N)
    r = np.empty(N)
    placed = 0
    for idx in order:
        res = _place_circle(W, H, cx[:placed], cy[:placed], r[:placed], rng,
                            n_samples=n_samples, n_refine=n_refine)
        if res is None:
            # no room: place a tiny circle at a random free-ish spot
            px = rng.uniform(0.1*W, 0.9*W)
            py = rng.uniform(0.1*H, 0.9*H)
            rad = _max_radius_at(px, py, cx[:placed], cy[:placed], r[:placed], W, H)
            if rad < 1e-4:
                rad = 1e-4
            cx[idx], cy[idx], r[idx] = px, py, rad
        else:
            cx[idx], cy[idx], r[idx] = res
        placed += 1
    return cx, cy, r

def _force_relax(cx, cy, r, W, H, iters=300, rng=None):
    """Repulsion-based relaxation: circles push apart, walls push in.
       Maximizes radii subject to non-overlap by growing r where space allows."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    if rng is None:
        rng = random.Random(12345)

    dt = 0.02
    for it in range(iters):
        if not _time_ok():
            break
        fx = np.zeros(N)
        fy = np.zeros(N)
        # pair repulsion: if too close, push apart; also allow growing radii
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                if d < 1e-9:
                    # separate randomly
                    ang = rng.uniform(0, 2*np.pi)
                    dx = np.cos(ang); dy = np.sin(ang); d = 1e-6
                overlap = (r[i] + r[j]) - d
                if overlap > 0:
                    # push apart proportionally
                    ux, uy = dx/d, dy/d
                    # each moves half
                    push = overlap * 0.5
                    fx[i] += ux * push
                    fy[i] += uy * push
                    fx[j] -= ux * push
                    fy[j] -= uy * push
                    # shrink a bit to guarantee separation
                    shrink = overlap * 0.15
                    r[i] = max(1e-4, r[i] - shrink)
                    r[j] = max(1e-4, r[j] - shrink)
                else:
                    # gap exists: try to grow both radii a little (use free space)
                    gap = -overlap
                    grow = min(gap * 0.1, 0.004)
                    r[i] += grow
                    r[j] += grow
        # wall forces: keep inside, and grow toward walls
        for i in range(N):
            # left
            if cx[i] - r[i] < 0:
                fx[i] += (r[i] - cx[i]) * 0.5
                r[i] = max(1e-4, r[i] * 0.98)
            else:
                # room to left: grow
                room = cx[i] - r[i]
                r[i] += min(room*0.1, 0.004)
            # right
            if (W - cx[i]) - r[i] < 0:
                fx[i] -= (cx[i] + r[i] - W) * 0.5
                r[i] = max(1e-4, r[i] * 0.98)
            else:
                room = (W - cx[i]) - r[i]
                r[i] += min(room*0.1, 0.004)
            # bottom
            if cy[i] - r[i] < 0:
                fy[i] += (r[i] - cy[i]) * 0.5
                r[i] = max(1e-4, r[i] * 0.98)
            else:
                room = cy[i] - r[i]
                r[i] += min(room*0.1, 0.004)
            # top
            if (H - cy[i]) - r[i] < 0:
                fy[i] -= (cy[i] + r[i] - H) * 0.5
                r[i] = max(1e-4, r[i] * 0.98)
            else:
                room = (H - cy[i]) - r[i]
                r[i] += min(room*0.1, 0.004)

        # apply forces
        cx += dt * fx
        cy += dt * fy
        # clamp inside
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        r = np.clip(r, 1e-4, min(W, H)/2)
    return cx, cy, r

def _validate_and_fix(cx, cy, r, W, H):
    """Enforce strict validity. Returns (valid, cx, cy, r, score)."""
    cx = cx.copy(); cy = cy.copy(); r = r.copy()
    # boundary fix
    for i in range(N):
        min_dist = min(cx[i], W-cx[i], cy[i], H-cy[i])
        if r[i] > min_dist:
            r[i] = max(1e-4, min_dist * 0.999)
    # overlap fix: iterative shrink
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i]-cx[j]; dy = cy[i]-cy[j]
                d = np.hypot(dx, dy)
                s = r[i]+r[j]
                if d < s:
                    if d < 1e-9:
                        # push apart by nudging positions
                        cx[i] += 0.001; cy[i] += 0.001
                        cx = np.clip(cx, 0, W); cy = np.clip(cy, 0, H)
                        d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                    factor = (d / s) * 0.999
                    r[i] = max(1e-4, r[i]*factor)
                    r[j] = max(1e-4, r[j]*factor)
                    changed = True
        if not changed:
            break
    # final validity check
    valid = True
    for i in range(N):
        if r[i] < 0:
            valid = False; break
        if cx[i] < r[i]-1e-8 or cx[i] > W-r[i]+1e-8:
            valid = False; break
        if cy[i] < r[i]-1e-8 or cy[i] > H-r[i]+1e-8:
            valid = False; break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i]-cx[j]; dy = cy[i]-cy[j]
                d2 = dx*dx+dy*dy
                s = r[i]+r[j]
                if d2 < s*s - 1e-8:
                    valid = False; break
            if not valid:
                break
    score = float(np.sum(r))
    return valid, cx, cy, r, score

def construct():
    best_score = 0.0
    best_out = None
    rng = random.Random(20240517)

    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2]

    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        if H <= 0 or W <= 0:
            continue

        # a few restarts with different insertion orders
        n_restarts = 4
        for restart in range(n_restarts):
            if not _time_ok():
                break
            order = list(range(N))
            if restart == 0:
                # default order
                pass
            else:
                rng.shuffle(order)

            cx, cy, r = _build_packing(W, H, order, rng, n_samples=250, n_refine=90)
            # relax
            cx, cy, r = _force_relax(cx, cy, r, W, H, iters=250, rng=rng)
            valid, cxf, cyf, rf, score = _validate_and_fix(cx, cy, r, W, H)
            if valid and score > best_score:
                best_score = score
                best_out = np.column_stack([cxf, cyf, rf])

    # fallback: guaranteed-valid tiny circles
    if best_out is None or best_score <= 0:
        W, H = 1.0, 1.0
        cx = np.linspace(0.15, 0.85, N)
        cy = np.full(N, 0.5)
        # spread vertically a bit
        cy = np.linspace(0.2, 0.8, N)
        r = np.full(N, 0.02)
        valid, cxf, cyf, rf, score = _validate_and_fix(cx, cy, r, W, H)
        if valid:
            best_out = np.column_stack([cxf, cyf, rf])
        else:
            best_out = np.column_stack([
                np.linspace(0.15, 0.85, N),
                np.linspace(0.2, 0.8, N),
                np.full(N, 0.01)
            ])

    return best_out
