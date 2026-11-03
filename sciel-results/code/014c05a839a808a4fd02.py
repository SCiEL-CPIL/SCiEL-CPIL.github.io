"""Circle packing in unit square; EVOLVE-BLOCK rewritten with best-fit local search + gradient polish."""
import numpy as np
import time

N = 26
DEADLINE = 150.0
SAFETY = 12.0          # stop this many seconds before the hard limit
START_TIME = time.time()


def _remaining():
    return DEADLINE - SAFETY - (time.time() - START_TIME)


def _max_radius_for_circle(i, pos, rad):
    """Max radius for circle i given positions pos (n,2) and radii rad (n,)."""
    xi, yi = pos[i]
    r = min(xi, yi, 1.0 - xi, 1.0 - yi)
    for j in range(N):
        if j == i:
            continue
        d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
        r = min(r, d - rad[j])
    return max(r, 0.0)


def _eval_packing(pos, rad):
    """Return (sum_radii, valid)."""
    for i in range(N):
        if rad[i] < -1e-9 or pos[i, 0] < -1e-9 or pos[i, 1] < -1e-9 \
                or pos[i, 0] > 1.0 + 1e-9 or pos[i, 1] > 1.0 + 1e-9:
            return 0.0, False
        if pos[i, 0] + rad[i] > 1.0 + 1e-7 or pos[i, 1] + rad[i] > 1.0 + 1e-7 \
                or pos[i, 0] - rad[i] < -1e-7 or pos[i, 1] - rad[i] < -1e-7:
            return 0.0, False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < rad[i] + rad[j] - 1e-7:
                return 0.0, False
    return float(np.sum(rad)), True


def _grid_init():
    """Dense structured grid starting layout."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 2))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            pos[idx, 0] = (i + 0.5) / side
            pos[idx, 1] = (j + 0.5) / side
            idx += 1
    rad = np.array([_max_radius_for_circle(i, pos, np.zeros(N)) for i in range(N)])
    return pos, rad


def _best_fit_local_search(pos, rad, iters, grid_res, deadline_check):
    """Discrete best-fit local search: move each circle to best of dense candidate grid."""
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_sum = float(np.sum(best_rad))
    step = 1.0 / grid_res
    # precompute candidate offsets
    for it in range(iters):
        if deadline_check():
            break
        improved = False
        for i in range(N):
            xi, yi = best_pos[i]
            cur_r = best_rad[i]
            best_r_here = cur_r
            best_pt_here = (xi, yi)
            # candidate positions in a neighborhood
            lo_x = max(0.0, xi - 2.0 * step)
            hi_x = min(1.0, xi + 2.0 * step)
            lo_y = max(0.0, yi - 2.0 * step)
            hi_y = min(1.0, yi + 2.0 * step)
            nx = int(np.ceil((hi_x - lo_x) / step)) + 1
            ny = int(np.ceil((hi_y - lo_y) / step)) + 1
            for a in range(nx):
                cx = lo_x + a * step
                if cx < 0.0:
                    cx = 0.0
                if cx > 1.0:
                    cx = 1.0
                for b in range(ny):
                    cy = lo_y + b * step
                    if cy < 0.0:
                        cy = 0.0
                    if cy > 1.0:
                        cy = 1.0
                    # compute max radius at this candidate
                    r = min(cx, cy, 1.0 - cx, 1.0 - cy)
                    for j in range(N):
                        if j == i:
                            continue
                        d = np.hypot(cx - best_pos[j, 0], cy - best_pos[j, 1])
                        r = min(r, d - best_rad[j])
                    if r > best_r_here:
                        best_r_here = r
                        best_pt_here = (cx, cy)
            if best_r_here > cur_r + 1e-12:
                best_pos[i, 0], best_pos[i, 1] = best_pt_here
                best_rad[i] = best_r_here
                improved = True
        if not improved:
            break
        new_sum = float(np.sum(best_rad))
        if new_sum > best_sum:
            best_sum = new_sum
    return best_pos, best_rad, best_sum


def _gradient_polish(pos, rad, deadline_check):
    """Polish radii by repeatedly recomputing max radius (radii depend on positions)."""
    pos = pos.copy()
    rad = rad.copy()
    for _ in range(50):
        if deadline_check():
            break
        changed = False
        for i in range(N):
            new_r = _max_radius_for_circle(i, pos, rad)
            if abs(new_r - rad[i]) > 1e-12:
                rad[i] = new_r
                changed = True
        if not changed:
            break
    return pos, rad


def _smooth_objective(pos, rad, gamma=80.0):
    """Penalized smooth objective: sum radii - penalty for violations."""
    obj = 0.0
    # wall violations
    for i in range(N):
        for v in (pos[i, 0] - rad[i], pos[i, 1] - rad[i],
                  1.0 - pos[i, 0] - rad[i], 1.0 - pos[i, 1] - rad[i]):
            if v < 0:
                obj -= gamma * v * v
    # overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            v = d - rad[i] - rad[j]
            if v < 0:
                obj -= gamma * v * v
    return obj


def construct():
    deadline_check = lambda: _remaining() < 0.0

    # Start from grid
    pos, rad = _grid_init()
    best_pos, best_rad, best_sum = pos.copy(), rad.copy(), float(np.sum(rad))

    # Best-fit local search with dense grid
    pos, rad, s = _best_fit_local_search(pos, rad, iters=30, grid_res=121,
                                         deadline_check=deadline_check)
    if s > best_sum:
        best_pos, best_rad, best_sum = pos.copy(), rad.copy(), s

    # Gradient polish
    pos, rad = _gradient_polish(pos, rad, deadline_check)
    s = float(np.sum(rad))
    if s > best_sum:
        best_pos, best_rad, best_sum = pos.copy(), rad.copy(), s

    # Another local search round from polished positions
    if _remaining() > 5.0:
        pos, rad, s = _best_fit_local_search(pos, rad, iters=15, grid_res=121,
                                             deadline_check=deadline_check)
        if s > best_sum:
            best_pos, best_rad, best_sum = pos.copy(), rad.copy(), s
        pos, rad = _gradient_polish(pos, rad, deadline_check)
        s = float(np.sum(rad))
        if s > best_sum:
            best_pos, best_rad, best_sum = pos.copy(), rad.copy(), s

    # Final validation & clip
    out = np.zeros((N, 3))
    for i in range(N):
        x = float(np.clip(best_pos[i, 0], 0.0, 1.0))
        y = float(np.clip(best_pos[i, 1], 0.0, 1.0))
        r = float(best_rad[i])
        # ensure inside
        r = min(r, x, y, 1.0 - x, 1.0 - y)
        r = max(r, 0.0)
        out[i] = (x, y, r)

    # verify validity; if invalid, fall back to safe grid
    _, valid = _eval_packing(out[:, :2], out[:, 2])
    if not valid:
        # shrink radii slightly to guarantee validity
        for i in range(N):
            out[i, 2] *= 0.98
        _, valid2 = _eval_packing(out[:, :2], out[:, 2])
        if not valid2:
            # ultimate fallback: trivial grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            out = np.array(cells[:N])

    return out
