"""Voronoi-tessellation circle packing: radii = soft-min distance from each seed
to its Voronoi cell boundary, optimized by gradient ascent on the sum of radii.
Disjoint interiors are guaranteed by construction (r_i <= dist to any other circle).
"""
import time
import numpy as np

N = 26
_DEADLINE = time.time() + 135.0   # safety margin well under the 150 s kill

# soft-min parameters (tuned for a good balance of smoothness / accuracy)
_TAU = 0.050      # soft-min temperature (smaller -> sharper)
_EPS = 1e-12
_WB = 3.0         # boundary weight (boundary is a hard constraint)
_WR = 1.0         # radius weight


def _soft_min(x, tau):
    """Smooth approximation of min(x). Returns (value, grad) with grad.sum()==1."""
    m = np.min(x)
    z = (x - m) / tau
    e = np.exp(-z)
    e = np.clip(e, 1e-300, None)
    s = np.sum(e)
    w = e / s
    return m + tau * np.log(s), w


def _voronoi_terms(pts):
    """For each seed i, the distance to its Voronoi cell boundary.
    Returns (vals (n,), grads (n, n), wall (n,), wall_grads (n, n)).
    vals[i] = soft_min over j!=i of d_ij/2, and over the 4 walls.
    """
    p = pts
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    d2 = 0.5 * dist
    g = 0.5 * d / dist          # grad of d_ij/2 wrt p[i]
    np.fill_diagonal(g[:, :, 0], 0.0)
    np.fill_diagonal(g[:, :, 1], 0.0)

    vals = np.empty(N)
    grads = np.zeros((N, N, 2))
    for i in range(N):
        v, w = _soft_min(d2[i], _TAU)
        vals[i] = v
        grads[i] = w[:, None] * g[i]

    # boundary (walls of the unit square)
    distb = np.minimum(p, 1.0 - p).min(axis=1)
    gb = np.zeros((N, N, 2))
    for i in range(N):
        drow = p[i, 0]
        dcol = 1.0 - p[i, 0]
        dtop = p[i, 1]
        dbot = 1.0 - p[i, 1]
        v, w = _soft_min(np.array([drow, dcol, dtop, dbot]), _TAU)
        vals[i] = min(vals[i], v)
        grads[i] += w[0] * np.array([-0.5, 0.0]) + w[1] * np.array([0.5, 0.0])
        grads[i] += w[2] * np.array([0.0, -0.5]) + w[3] * np.array([0.0, 0.5])

    return vals, grads, distb, gb


def _objective(pts, tau):
    """Returns (score, grad). score = sum_i soft_min(r_i, wall_i) with boundary
    penalty folded in so that boundary violations are strongly discouraged."""
    vals, grads, distb, gb = _voronoi_terms(pts)
    # combine radius term and boundary term with a weighted soft-min
    score = np.empty(N)
    grad = np.zeros((N, 2))
    for i in range(N):
        r = vals[i]
        b = distb[i]
        # weighted soft-min of (r, b) with weights (WR, WB)
        pair = np.array([r, b])
        m = np.min(pair)
        z = (pair - m) / tau
        e = np.clip(np.exp(-z), 1e-300, None)
        w = e / np.sum(e)
        score[i] = m + tau * np.log(np.sum(e))
        # grads: r-grad is grads[i] (already for r), boundary-grad is gb[i]
        grad[i] = w[0] * grads[i] + w[1] * gb[i]
    return np.sum(score), grad


def _is_valid(pts, r):
    """Check disjoint interiors and containment with a small safety margin."""
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    if np.min(d) < (r[:, None] + r[None, :]).max() - 1e-9:
        # pairwise overlap test
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < r[i] + r[j] - 1e-9:
                    return False
    if np.any(pts < -1e-9) or np.any(pts > 1.0 + 1e-9):
        return False
    if np.any(r > np.minimum(pts, 1.0 - pts).min(axis=1) + 1e-9):
        return False
    return True


def _make_grid_initial():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for j in range(side):
        for i in range(side):
            cells.append((0.5 / side + i * (1.0 / side),
                          0.5 / side + j * (1.0 / side)))
    return np.array(cells[:N], dtype=float)


def _project(pts):
    return np.clip(pts, 0.0, 1.0)


def _sum_radii(pts):
    vals, _, _, _ = _voronoi_terms(pts)
    return vals


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best_pts = _make_grid_initial()
    best_r = _sum_radii(best_pts)
    best_score = float(np.sum(best_r))

    def elapsed_ok():
        return (time.time() - t0) < _DEADLINE

    # ---- main gradient ascent with adaptive step + occasional restarts ----
    step = 0.01
    it = 0
    while elapsed_ok() and it < 4000:
        it += 1
        val, grad = _objective(best_pts, _TAU)
        # normalize gradient for stable steps
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            ghat = grad / gnorm
        else:
            ghat = grad
        # try a step; accept if the raw sum of radii improves
        cand = _project(best_pts + step * ghat)
        cand_r = _sum_radii(cand)
        cand_score = float(np.sum(cand_r))
        if cand_score > best_score + 1e-12:
            best_pts = cand
            best_score = cand_score
            best_r = cand_r
            step = min(step * 1.03, 0.05)   # increase step on success
        else:
            # try a smaller step
            cand2 = _project(best_pts + step * 0.5 * ghat)
            cand2_r = _sum_radii(cand2)
            cand2_score = float(np.sum(cand2_r))
            if cand2_score > best_score + 1e-12:
                best_pts = cand2
                best_score = cand2_score
                best_r = cand2_r
            else:
                step *= 0.7
                if step < 1e-5:
                    # stuck: do a random nudge (restart-ish)
                    best_pts = _project(best_pts + 0.02 * (rng.random((N, 2)) - 0.5))
                    best_r = _sum_radii(best_pts)
                    best_score = float(np.sum(best_r))
                    step = 0.01

        # periodic restart from a perturbed copy of best to escape local minima
        if it % 300 == 0:
            pert = _project(best_pts + 0.05 * (rng.random((N, 2)) - 0.5))
            pr = _sum_radii(pert)
            ps = float(np.sum(pr))
            if ps > best_score:
                best_pts, best_r, best_score = pert, pr, ps
            step = 0.01

    # ---- final high-precision polish: tighten tau, more fine steps ----
    if elapsed_ok():
        fine_pts = best_pts.copy()
        fine_step = 0.002
        for _ in range(600):
            if not elapsed_ok():
                break
            val, grad = _objective(fine_pts, 0.01)   # sharper soft-min
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-12:
                ghat = grad / gnorm
            else:
                break
            cand = _project(fine_pts + fine_step * ghat)
            cand_r = _sum_radii(cand)
            cand_score = float(np.sum(cand_r))
            if cand_score > best_score + 1e-12:
                best_pts, best_r, best_score = cand, cand_r, cand_score
                fine_step = min(fine_step * 1.02, 0.01)
            else:
                fine_step *= 0.7
                if fine_step < 1e-6:
                    break

    # ---- build radii from the final Voronoi distances (exact min) ----
    p = best_pts
    d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    r = 0.5 * np.min(d, axis=1)
    r = np.minimum(r, np.minimum(p, 1.0 - p).min(axis=1))
    r = np.maximum(r, 1e-6)

    # final safety: if anything is invalid, fall back to the grid seed
    if not _is_valid(p, r):
        p = _make_grid_initial()
        d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        r = 0.5 * np.min(d, axis=1)
        r = np.minimum(r, np.minimum(p, 1.0 - p).min(axis=1))
        r = np.maximum(r, 1e-6)

    return np.column_stack([p[:, 0], p[:, 1], r])
