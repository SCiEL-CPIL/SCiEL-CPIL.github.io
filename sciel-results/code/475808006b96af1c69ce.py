"""Structural change: combinatorial cage initializations + multi-stage penalty BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_EPS = 1e-12


def _diameter(pts):
    n = len(pts)
    d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = max(d, dx * dx + dy * dy)
    return math.sqrt(d)


def _log_product(pts):
    """Sum of log distances over ordered pairs (i != j)."""
    n = len(pts)
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < _EPS:
                return -1e18
            s += math.log(d)
    return s


def _score_from_log_product_and_diameter(log_prod, diam):
    """Score = Delta / n^n where Delta = product over ordered pairs after scaling to diam=2.
    Product over ordered pairs = (product over unordered pairs)^2.
    After scaling by 2/diam, each distance scales by 2/diam.
    Number of ordered pairs = n*(n-1).
    """
    n = 8
    # product over ordered pairs after scaling:
    # log = log_prod + n*(n-1)*log(2/diam)
    if diam < _EPS:
        return 0.0
    log_scaled = log_prod + n * (n - 1) * math.log(2.0 / diam)
    # Delta / n^n = exp(log_scaled) / n^n
    log_score = log_scaled - n * n * math.log(n)
    return math.exp(log_score)


def _score(pts):
    diam = _diameter(pts)
    if diam < _EPS:
        return 0.0
    log_prod = _log_product(pts)
    return _score_from_log_product_and_diameter(log_prod, diam)


def _objective(x, penalty_weight, diam_target=2.0):
    """Minimize negative log product with penalty for diameter != target."""
    n = 8
    pts = x.reshape(n, 2)
    # diameter
    d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = max(d, dx * dx + dy * dy)
    diam = math.sqrt(d) if d > 0 else _EPS
    # log product
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dd = dx * dx + dy * dy
            if dd < _EPS:
                return 1e18
            s += 0.5 * math.log(dd)
    # log_prod (unordered) = s; ordered = 2*s
    log_prod_ordered = 2.0 * s
    # penalty: want diam close to diam_target
    pen = penalty_weight * (diam - diam_target) ** 2
    return -log_prod_ordered + pen


def _refine(pts0, time_budget):
    """Multi-stage penalty BFGS refinement."""
    n = 8
    x0 = np.array(pts0, dtype=float).reshape(n * 2)
    # Normalize to diameter 2 first
    diam = _diameter(pts0)
    if diam > _EPS:
        x0 = x0 * (2.0 / diam)
    best_x = x0.copy()
    best_score = _score(best_x.reshape(n, 2).tolist())

    # Multi-stage: start with small penalty, increase
    penalties = [0.01, 0.1, 1.0, 10.0, 100.0]
    t_start = time.time()
    for pw in penalties:
        if time.time() - t_start > time_budget:
            break
        try:
            res = minimize(
                _objective, x0, args=(pw,),
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if res.success or res.fun < 1e17:
                cand = res.x.reshape(n, 2).tolist()
                sc = _score(cand)
                if sc > best_score:
                    best_score = sc
                    best_x = res.x.copy()
                x0 = res.x.copy()
        except Exception:
            break
        if time.time() - t_start > time_budget:
            break
    return best_x.reshape(n, 2).tolist(), best_score


def _make_cage_init(mode, rng):
    """Generate a 'cage' topology: two concentric squares (or similar)."""
    n = 8
    if mode == 0:
        # Two concentric squares, outer radius 1, inner radius r
        r = rng.uniform(0.2, 0.8)
        outer = []
        inner = []
        for k in range(4):
            ang = 2 * math.pi * k / 4
            outer.append([math.cos(ang), math.sin(ang)])
            iang = ang + math.pi / 4  # offset inner square by 45 deg
            inner.append([r * math.cos(iang), r * math.sin(iang)])
        pts = outer + inner
    elif mode == 1:
        # Two concentric squares, same orientation
        r = rng.uniform(0.2, 0.8)
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([math.cos(ang), math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([r * math.cos(ang), r * math.sin(ang)])
    elif mode == 2:
        # Regular octagon perturbed
        pts = []
        for k in range(8):
            ang = 2 * math.pi * k / 8 + rng.uniform(-0.15, 0.15)
            rad = 1.0 + rng.uniform(-0.2, 0.2)
            pts.append([rad * math.cos(ang), rad * math.sin(ang)])
    elif mode == 3:
        # Concentric squares with random inner radius and rotation
        r = rng.uniform(0.1, 0.9)
        rot = rng.uniform(0, math.pi / 4)
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([math.cos(ang), math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4 + rot
            pts.append([r * math.cos(ang), r * math.sin(ang)])
    else:
        # Random
        pts = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(8)]
    return pts


def construct():
    t_start = time.time()
    time_budget = _DEADLINE - 10.0  # safety margin
    rng = np.random.default_rng(42)

    # Regular octagon as baseline
    octagon = [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]
    best_pts = octagon
    best_score = _score(octagon)

    # Try cage initializations
    n_modes = 5
    n_trials_per_mode = 3
    total_trials = n_modes * n_trials_per_mode
    per_trial_budget = time_budget / (total_trials + 2)

    for mode in range(n_modes):
        for trial in range(n_trials_per_mode):
            if time.time() - t_start > _DEADLINE - 10.0:
                return best_pts
            init = _make_cage_init(mode, rng)
            # Ensure distinct points
            ok = True
            for i in range(8):
                for j in range(i + 1, 8):
                    dx = init[i][0] - init[j][0]
                    dy = init[i][1] - init[j][1]
                    if dx * dx + dy * dy < 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            try:
                refined, sc = _refine(init, per_trial_budget)
                if sc > best_score:
                    best_score = sc
                    best_pts = refined
            except Exception:
                pass

    # Final polish with more iterations if time allows
    if time.time() - t_start < _DEADLINE - 15.0:
        remaining = _DEADLINE - 15.0 - (time.time() - t_start)
        try:
            refined, sc = _refine(best_pts, remaining)
            if sc > best_score:
                best_score = sc
                best_pts = refined
        except Exception:
            pass

    # Ensure coordinates are within bounds
    result = []
    for p in best_pts:
        x = float(p[0])
        y = float(p[1])
        if abs(x) >= 1e6:
            x = math.copysign(1e6 - 1.0, x)
        if abs(y) >= 1e6:
            y = math.copysign(1e6 - 1.0, y)
        result.append([x, y])
    return result
