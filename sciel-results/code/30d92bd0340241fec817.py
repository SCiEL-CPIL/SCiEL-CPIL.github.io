# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill

# Sofa grid points (fixed)
_XI = np.arange(-3.0, 1.5, 0.01)
_YJ = np.arange(-1.5, 1.5, 0.01)
_PX, _PY = np.meshgrid(_XI, _YJ, indexing="ij")
_PX = _PX.ravel()
_PY = _PY.ravel()
_NP = _PX.size


def _corridor_mask(a_deg):
    """Boolean mask over sofa points inside the corridor at rotation a (deg),
    with inner corner at origin. Vectorized over all sofa points."""
    rad = np.deg2rad(a_deg)
    c, s = np.cos(rad), np.sin(rad)
    # R(-a) applied to (x,y): u = c*x + s*y ; v = -s*x + c*y
    u = c * _PX + s * _PY
    v = -s * _PX + c * _PY
    m1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    m2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return m1 | m2


def _build_path(pivot):
    """Build full densified path (x,y,a) rows given a pivot (px,py).
    Evaluator prepends (0,0,0) and appends (-2,0,90)."""
    px, py = pivot
    rad = np.deg2rad(90.0)
    c, s = np.cos(rad), np.sin(rad)
    # end inner corner = pivot + R(90)(-pivot) = (-py, px)
    ex, ey = -py, px
    # require end inner corner == (-2, 0)
    # => px = 0, py = 2
    return px, py


def _full_path_rows(px, py):
    """Return the complete sequence of (x,y,a) rows the evaluator will use:
    start (0,0,0), our pivot path (a from 0..90), end (-2,0,90)."""
    # pivot path: inner corner = pivot + R(a)(-pivot)
    # We sample a over 0..90. Evaluator also densifies between consecutive rows.
    # To be safe and faithful, we just return a dense pivot path; the evaluator
    # will prepend start and append end and densify. We return rows for a=0..90
    # but note evaluator adds its own start/end. We'll return the pivot rows for
    # a in a small set; evaluator densifies. To maximize fidelity we return many.
    a = np.linspace(0.0, 90.0, 181)  # 0.5 deg steps
    rad = np.deg2rad(a)
    c, s = np.cos(rad), np.sin(rad)
    # inner corner = pivot + R(a)(-pivot)
    dx = -px
    dy = -py
    # R(a)(dx,dy) = (c*dx - s*dy, s*dx + c*dy)
    rx = c * dx - s * dy
    ry = s * dx + c * dy
    xs = px + rx
    ys = py + ry
    return np.column_stack([xs, ys, a])


def _evaluate_path(rows):
    """Compute sofa area given a full (n,3) path (x,y,a) rows, with linear
    densification between consecutive rows as the evaluator does. Returns area."""
    # Prepend start, append end (as evaluator does)
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    full = np.vstack([start, rows, end])

    # Densify: between consecutive rows add linear interpolations in
    # m = max(1, ceil(|a2-a1|)) equal steps.
    dens = [full[0]]
    for i in range(len(full) - 1):
        a1 = full[i, 2]
        a2 = full[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        # m equal steps => m+1 points including endpoints; we already have start,
        # add intermediate and the endpoint (endpoint becomes next iteration start).
        t = np.linspace(0.0, 1.0, m + 1)
        seg = full[i] + t[:, None] * (full[i + 1] - full[i])
        dens.extend(seg[1:].tolist())  # skip the start (already added)
    dens = np.array(dens)

    # For each placement, compute mask; sofa = intersection of all masks.
    # To be efficient, accumulate the intersection incrementally and stop early
    # when it becomes empty.
    keep = np.ones(_NP, dtype=bool)
    for i in range(dens.shape[0]):
        x, y, a = dens[i]
        rad = np.deg2rad(a)
        c, s = np.cos(rad), np.sin(rad)
        u = c * (_PX - x) + s * (_PY - y)
        v = -s * (_PX - x) + c * (_PY - y)
        m1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        m2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = m1 | m2
        keep &= mask
        if not keep.any():
            break
    return keep.sum() * 0.0001


def moving_sofa_2d() -> np.ndarray:
    # The pivot must satisfy end inner corner = (-2, 0), which forces px=0, py=2.
    # So the "2D optimization" collapses to a single valid pivot. We still run a
    # small search over nearby pivots to be robust, but the constrained optimum
    # is at (0, 2). We return the pivot path for (0,2) as the baseline, and if
    # time permits, refine the number of waypoints / check neighbors.

    best_rows = _full_path_rows(0.0, 2.0)
    best_area = _evaluate_path(best_rows)

    # Coarse-to-fine: try a few nearby pivots and waypoint counts to see if any
    # improves. (The hard constraint pins the pivot, but we verify numerically.)
    candidates = [(0.0, 2.0)]
    for dpx in (-0.05, 0.0, 0.05):
        for dpy in (-0.05, 0.0, 0.05):
            candidates.append((0.0 + dpx, 2.0 + dpy))

    for px, py in candidates:
        if time.time() > _DEADLINE:
            break
        for nway in (181, 361, 90):
            if time.time() > _DEADLINE:
                break
            a = np.linspace(0.0, 90.0, nway)
            rad = np.deg2rad(a)
            c, s = np.cos(rad), np.sin(rad)
            rx = c * (-px) - s * (-py)
            ry = s * (-px) + c * (-py)
            xs = px + rx
            ys = py + ry
            rows = np.column_stack([xs, ys, a])
            area = _evaluate_path(rows)
            if area > best_area:
                best_area = area
                best_rows = rows

    return best_rows


# EVOLVE-BLOCK-END
