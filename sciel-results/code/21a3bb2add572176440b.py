"""Max product of pairwise distances for 8 planar points at fixed diameter.
Score = Delta / 8^8, Delta = product over ordered pairs of distances after
scaling the set to diameter 2. Regular octagon scores exactly 1.0.
Strategy: D4-symmetric parameterization (two concentric squares, 3 params),
global grid + local refinement, then a full 16-DOF polish.
"""
import math
import time

import numpy as np
from scipy.optimize import minimize

_START = time.time()
_LIMIT = 150.0          # wall-clock safety margin (evaluator kills at ~150s)


def _left():
    return _LIMIT - (time.time() - _START)


def _logscore_pts(P):
    """log of the raw score for an (8,2) array; -inf-ish if degenerate."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    diam = d.max()
    if diam <= 1e-12:
        return -1e300
    ds = d / (diam / 2.0)
    m = ~np.eye(n, dtype=bool)
    v = ds[m]
    if (v <= 1e-12).any():
        return -1e300
    return float(np.log(v).sum() - n * math.log(n))


def _score_pts(P):
    ls = _logscore_pts(P)
    if ls < -1e299:
        return 0.0
    return math.exp(ls)


def _pts3(R, r, theta):
    """Two concentric squares: outer radius R rotated by theta, inner radius r."""
    R = max(R, 1e-4)
    r = max(r, 1e-4)
    P = np.empty((8, 2))
    for k in range(4):
        a = theta + k * math.pi / 2.0
        P[2 * k] = [R * math.cos(a), R * math.sin(a)]
        b = k * math.pi / 2.0
        P[2 * k + 1] = [r * math.cos(b), r * math.sin(b)]
    return P


def _logscore3(x):
    return _logscore_pts(_pts3(x[0], x[1], x[2]))


def _valid(P):
    P = np.asarray(P, dtype=float)
    if P.shape != (8, 2) or not np.all(np.isfinite(P)):
        return False
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    return d.min() > 1e-6 and d.max() > 1e-6


def _regular_octagon():
    return [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)]
            for k in range(8)]


def construct():
    best_P = np.array(_regular_octagon(), dtype=float)
    best_ls = _logscore_pts(best_P)

    # ---------- Stage 1: coarse global grid over the 3 D4 parameters ----------
    nR = nr = nth = 40
    Rs = np.linspace(0.10, 1.70, nR)
    rs = np.linspace(0.10, 1.70, nr)
    ths = np.linspace(0.0, math.pi / 2.0, nth)
    grid_best = []
    for R in Rs:
        for r in rs:
            for th in ths:
                if _left() < 40.0:
                    break
                ls = _logscore3([R, r, th])
                if ls > best_ls:
                    best_ls = ls
                    best_P = _pts3(R, r, th)
                grid_best.append((ls, R, r, th))
    grid_best.sort(key=lambda t: -t[0])

    # ---------- Stage 2: local 3-param refinement from top grid starts ----------
    starts = [t[1:] for t in grid_best[:8]]
    # also seed with the regular-octagon-equivalent (R=r=1, th=pi/4)
    starts.append((1.0, 1.0, math.pi / 4.0))
    for s in starts:
        if _left() < 30.0:
            break
        try:
            res = minimize(lambda x: -_logscore3(x), np.array(s, dtype=float),
                           method="Nelder-Mead",
                           options={"maxiter": 2500, "xatol": 1e-9, "fatol": 1e-12})
            ls = _logscore3(res.x)
            if ls > best_ls:
                best_ls = ls
                best_P = _pts3(*res.x)
        except Exception:
            pass

    # ---------- Stage 3: fine grid around the current best 3-param point ----------
    if _left() > 25.0:
        R0, r0, th0 = best_P  # not directly; recompute from stored params
    # recover params by re-searching a small window around best_P via 3-param fit:
    # simpler: run a fine grid in a window around the last refined params.
    if _left() > 25.0:
        # find the best 3-param vector among refined candidates by quick local grid
        Rg = np.linspace(max(0.05, R0 - 0.25) if False else 0.05, 1.75, 26)
        rg = np.linspace(0.05, 1.75, 26)
        tg = np.linspace(0.0, math.pi / 2.0, 26)
        # focus window around the best start we have
        # (use the top refined start as center)
        center = None
        for s in starts:
            ls = _logscore3(s)
            if center is None or ls > _logscore3(center):
                center = s
        if center is not None:
            cR, cr, cth = center
            Rg = np.linspace(max(0.02, cR - 0.18), min(1.78, cR + 0.18), 26)
            rg = np.linspace(max(0.02, cr - 0.18), min(1.78, cr + 0.18), 26)
            tg = np.linspace(0.0, math.pi / 2.0, 26)
            for R in Rg:
                for r in rg:
                    for th in tg:
                        if _left() < 20.0:
                            break
                        ls = _logscore3([R, r, th])
                        if ls > best_ls:
                            best_ls = ls
                            best_P = _pts3(R, r, th)

    # ---------- Stage 4: full 16-DOF polish (escape D4 symmetry if useful) ----------
    def _neg(x):
        return -_logscore_pts(x.reshape(8, 2))

    if _left() > 25.0 and _valid(best_P):
        for trial in range(3):
            if _left() < 20.0:
                break
            x0 = best_P.copy().ravel()
            if trial > 0:
                x0 = x0 + 0.02 * np.random.RandomState(7 + trial).randn(16)
            try:
                res = minimize(_neg, x0, method="Nelder-Mead",
                               options={"maxiter": 1200, "xatol": 1e-8, "fatol": 1e-11})
                cand = res.x.reshape(8, 2)
                if _valid(cand):
                    ls = _logscore_pts(cand)
                    if ls > best_ls:
                        best_ls = ls
                        best_P = cand
            except Exception:
                pass

    # ---------- Final: pick the best valid candidate ----------
    candidates = [np.array(_regular_octagon(), dtype=float), best_P]
    final = None
    final_ls = -1e300
    for c in candidates:
        if _valid(c):
            ls = _logscore_pts(c)
            if ls > final_ls:
                final_ls = ls
                final = c
    if final is None:
        final = np.array(_regular_octagon(), dtype=float)

    # normalize to diameter 2 for clean output (score is scale-invariant)
    d = np.sqrt(((final[:, None, :] - final[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    diam = d.max()
    if diam > 1e-12:
        final = final / (diam / 2.0)

    return [[float(x), float(y)] for x, y in final]
