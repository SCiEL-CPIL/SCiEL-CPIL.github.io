import time
import math
import numpy as np

N = 12

# ----------------------------------------------------------------------
# Geometry helpers (regular hexagons, flat-top orientation, angle 0)
# ----------------------------------------------------------------------
# Inward unit normals for a flat-top regular hexagon (apothem along +x)
# 3 axes: x, 120deg, 240deg
_AX = np.array([1.0, 0.0])
_AX2 = np.array([math.cos(math.radians(120)), math.sin(math.radians(120))])
_AX3 = np.array([math.cos(math.radians(240)), math.sin(math.radians(240))])
_AXES = np.array([_AX, _AX2, _AX3])          # shape (3,2)
# outward normals for the Minkowski-sum (collision) hexagon of two unit hexes
_OUT = np.array([_AX, _AX2, _AX3])


def _hex_inward_max(p):
    """max over inward normals n of (n . p).  p: (2,) or (M,2)."""
    return np.max(p @ _AXES.T, axis=1)


def _sdf_outer(p, r):
    """Signed distance to a centered flat-top hexagon of apothem r.
    Positive = outside.  p: (2,) or (M,2)."""
    return _hex_inward_max(p) - r


def _collision_violation(d):
    """For distance d between two same-orientation unit-hex centers,
    the overlap amount (Minkowski hex of circumradius 2 => apothem 2*sqrt3/2).
    Returns how far d is below the required 2 (0 if separated)."""
    # Minkowski sum of two unit flat-top hexes is a flat-top hex with apothem = 2*(sqrt3/2)=sqrt3
    # containment: max(n.d) <= apothem.  violation = max(n.d) - apothem
    return np.max(d @ _OUT.T, axis=1) - math.sqrt(3)


# ----------------------------------------------------------------------
# Objective: minimize outer side S subject to feasibility, with smooth penalty
# ----------------------------------------------------------------------
def _objective(z, S_target, lam):
    # z = [cx1..cx12, cy1..cy12, ox, oy, S]
    cx = z[0:12]
    cy = z[12:24]
    ox = z[24]
    oy = z[25]
    S = z[26]
    S = max(S, 1.0)
    r = S * math.sqrt(3) / 2.0          # outer apothem

    # relative positions of inner centers w.r.t. outer center
    px = cx - ox
    py = cy - oy
    p = np.column_stack([px, py])

    # 1) containment: each inner hex must fit with clearance 1 (apothem 1)
    #    max(n . (c - o)) <= r - 1
    cont_viol = np.max(p @ _AXES.T, axis=1) - (r - 1.0)
    cont_pen = np.sum(np.maximum(cont_viol, 0.0) ** 2)

    # 2) pairwise separation: Minkowski hex violation
    viol = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            v = max(abs(dx), abs(dx) * 0.5 + math.sqrt(3) / 2 * abs(dy))  # hex metric
            # apothem of Minkowski hex = sqrt(3); required v <= sqrt(3)
            ov = v - math.sqrt(3)
            if ov > 0:
                viol += ov * ov
    sep_pen = viol

    # 3) encourage S near/under target
    size_pen = (S - S_target) ** 2 if S > S_target else 0.0

    return S + lam * (cont_pen + sep_pen) + 0.5 * size_pen


def _feasibility(z):
    """Return (max_cont_viol, max_sep_viol) for a candidate z."""
    cx = z[0:12]
    cy = z[12:24]
    ox, oy, S = z[24], z[25], z[26]
    r = S * math.sqrt(3) / 2.0
    p = np.column_stack([cx - ox, cy - oy])
    cont_viol = np.max(np.max(p @ _AXES.T, axis=1) - (r - 1.0), axis=0)
    max_sep = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            v = max(abs(dx), abs(dx) * 0.5 + math.sqrt(3) / 2 * abs(dy))
            max_sep = max(max_sep, v - math.sqrt(3))
    return cont_viol, max_sep


def _is_feasible(z, tol=1e-6):
    c, s = _feasibility(z)
    return (c <= tol) and (s <= tol)


# ----------------------------------------------------------------------
# Optimization pipeline: L-BFGS-B (smooth) then Nelder-Mead polish
# ----------------------------------------------------------------------
from scipy.optimize import minimize


def _optimize_once(z0, S_target, lam, deadline, best):
    # L-BFGS-B on the smooth objective
    res = minimize(_objective, z0, args=(S_target, lam), method='L-BFGS-B',
                   options={'maxiter': 200, 'maxfun': 4000,
                            'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 40})
    z = res.x
    if time.time() > deadline:
        return best
    # Nelder-Mead polish with a tight penalty to reach feasibility
    res2 = minimize(_objective, z, args=(S_target, lam * 4.0), method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9,
                             'adaptive': True})
    z = res2.x
    if time.time() > deadline:
        return best
    # update best if feasible and smaller
    if _is_feasible(z):
        if z[26] < best[0]:
            best = (z[26], z.copy())
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 140.0          # safety margin before 150s kill

    best = (float('inf'), None)

    # --- initial guesses: a hexagonal lattice of 12 cells inside a big hex ---
    # Use a triangular/hex lattice with spacing ~2.1 (just over the required 2)
    def _lattice_guess(S0):
        rng = np.random.default_rng(0)
        # 7 centers in a hexagonal cluster + 5 more -> 12
        pts = []
        # hex cluster of radius 1 (7 points)
        for a in range(-1, 2):
            for b in range(-1, 2):
                if abs(a + b) <= 1:
                    x = 2.05 * (a + 0.5 * b)
                    y = 2.05 * (math.sqrt(3) / 2 * b)
                    pts.append([x, y])
        # add 5 more around
        for k in range(5):
            ang = math.radians(60 * k)
            pts.append([4.2 * math.cos(ang), 4.2 * math.sin(ang)])
        pts = np.array(pts[:12])
        cx = pts[:, 0] + rng.normal(0, 0.1, 12)
        cy = pts[:, 1] + rng.normal(0, 0.1, 12)
        z = np.concatenate([cx, cy, [0.0, 0.0, [S0]]])
        return z

    # Annealing schedule: start with a generous S, shrink toward the optimum
    S_starts = [14.0, 12.0, 10.5, 9.5, 9.0, 8.5, 8.0]
    lam_starts = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]

    # Seed RNG for restarts
    rng = np.random.default_rng(1)

    # Phase 1: annealed optimization from lattice
    for S0, lam in zip(S_starts, lam_starts):
        if time.time() > deadline:
            break
        z0 = _lattice_guess(S0)
        best = _optimize_once(z0, S0, lam, deadline, best)

    # Phase 2: random restarts to escape local minima, targeting the best so far
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        S_target = best[0] * 0.97 if best[0] < float('inf') else 9.0
        lam = 200.0
        # random perturbation of the current best or a fresh lattice
        if best[1] is not None and rng.random() < 0.6:
            z0 = best[1] + rng.normal(0, 0.3, 27)
            z0[26] = S_target
        else:
            z0 = _lattice_guess(max(S_target * 1.1, 9.5))
        # add small random rotation-free jitter
        best = _optimize_once(z0, S_target, lam, deadline, best)
        if attempts > 200:
            break

    if best[1] is None:
        # fallback: return a guaranteed-feasible large placement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    z = best[1]
    inner = [[float(z[i]), float(z[i + 12]), 0.0] for i in range(12)]
    center = (float(z[24]), float(z[25]))
    side = float(z[26])
    return inner, center, side, 0.0
