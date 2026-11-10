# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_SQRT3 = np.sqrt(3.0)
_SQRT11 = np.sqrt(11.0)
_TOL = 1e-6
_TOL2 = 1e-12

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _pair_gap(A, B):
    """Max separation along any of the 12 edge normals. >0 => separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
    return best

def _outer_side_for_points(pts):
    """Smallest outer-hex side (center at origin) containing pts, over 60 deg."""
    best_s = np.inf
    for a_deg in np.linspace(0.0, 60.0, 361):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T
        apothem = proj.max().max()
        s = 2.0 * apothem / _SQRT3
        if s < best_s:
            best_s = s
    return best_s

def _all_valid(inner, outer, side):
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(side)):
        return False
    if side < _SQRT11 - 1e-9:
        return False
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            if _pair_gap(Vs[i], Vs[j]) <= _TOL:
                return False
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -_TOL):
                return False
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

# ---------------------------------------------------------------------------
# Vectorized objective for global optimization
# ---------------------------------------------------------------------------
def _compute_side(pts, s):
    """Outer-hex side (center origin) needed to contain pts, given rotation s."""
    a = np.deg2rad(s)
    th = a + np.deg2rad(60.0) * np.arange(6)
    normals = np.column_stack((np.cos(th), np.sin(th)))
    proj = pts @ normals.T
    apothem = proj.max().max()
    return 2.0 * apothem / _SQRT3

def _objective(x):
    """x = [x0..x10, y0..y10, a0..a10(deg), s_outer]. Minimize s_outer + penalties."""
    x0 = x[0:11]
    y0 = x[11:22]
    a0 = x[22:33]
    s = x[33]
    pts = np.column_stack((x0, y0))

    # outer size
    side = _compute_side(pts, s)
    val = side

    # separation penalties (vectorized per pair)
    Vs = [_hex_vertices(x0[i], y0[i], 1.0, a0[i]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m = _pair_gap(Vs[i], Vs[j])
            if m <= _TOL:
                viol = _TOL - m
                val += 1e4 * viol * viol
    return val

# ---------------------------------------------------------------------------
# Seed arrangement (hexagonal lattice, unit spacing)
# ---------------------------------------------------------------------------
def _seed_positions():
    return np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [-2.0, 0.0],
        [1.0, 1.732],
        [-1.0, 1.732],
        [1.0, -1.732],
        [-1.0, -1.732],
        [3.0, 1.732],
        [-3.0, 1.732],
        [3.0, -1.732],
        [-3.0, -1.732],
    ])

def _state_from_pos(pos, angles, s):
    return np.concatenate([pos[:, 0], pos[:, 1], angles, [s]])

# ---------------------------------------------------------------------------
def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 600.0

    base = _seed_positions()
    s0 = _outer_side_for_points(base)

    def _eval_state(state):
        x0 = state[0:11]
        y0 = state[11:22]
        a0 = state[22:33]
        pts = np.column_stack((x0, y0))
        s, a_out = _outer_side_for_points(pts)
        inner = np.column_stack((x0, y0, a0))
        outer = np.array([0.0, 0.0, a_out])
        if _all_valid(inner, outer, s):
            return 1.0 / s, inner, outer, s
        return None, inner, outer, s

    best_val = -np.inf
    best_state = None

    # evaluate seed
    seed_state = _state_from_pos(base, np.zeros(11), s0)
    val, inner, outer, s = _eval_state(seed_state)
    if val is not None and val > best_val:
        best_val = val
        best_state = (inner, outer, s)

    # global optimization via differential_evolution
    if time.time() < deadline - 120.0:
        rng = np.random.default_rng(7)
        # bounds: positions roughly within [-6,6], angles [-60,60], s [sqrt11, 10]
        bounds = []
        for _ in range(11):
            bounds.append((-6.0, 6.0))   # x
        for _ in range(11):
            bounds.append((-6.0, 6.0))   # y
        for _ in range(11):
            bounds.append((-60.0, 60.0)) # angle deg
        bounds.append((_SQRT11, 10.0))   # s outer

        # initial population around the seed
        pop = []
        for _ in range(20):
            p = base + rng.normal(0, 0.4, size=(11, 2))
            ang = rng.uniform(-30, 30, size=11)
            s = s0 + rng.normal(0, 0.2)
            s = np.clip(s, _SQRT11, 10.0)
            pop.append(_state_from_pos(p, ang, s))
        # ensure seed is in the population
        pop[0] = seed_state

        try:
            res = differential_evolution(
                _objective, bounds, x0=pop,
                seed=7, maxiter=200, popsize=20, tol=1e-8,
                mutation=(0.3, 1.0), recombination=0.7,
                polish=True, init='x0',
            )
            if res.x is not None:
                val, inner, outer, s = _eval_state(res.x)
                if val is not None and val > best_val:
                    best_val = val
                    best_state = (inner, outer, s)
        except Exception:
            pass

    # local refinement with L-BFGS-B on the best state found
    if best_state is not None and time.time() < deadline - 60.0:
        x0 = best_state[0][:, 0]
        y0 = best_state[0][:, 1]
        a0 = best_state[0][:, 2]
        s = best_state[2]
        start = _state_from_pos(np.column_stack((x0, y0)), a0, s)
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, start, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
            if res.x is not None:
                val, inner, outer, s2 = _eval_state(res.x)
                if val is not None and val > best_val:
                    best_val = val
                    best_state = (inner, outer, s2)
        except Exception:
            pass

    if best_state is None:
        # fallback loose packing
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0

    inner, outer, s = best_state
    return inner, outer, float(s)

# EVOLVE-BLOCK-END
