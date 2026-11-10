# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve

_T0 = time.time()
_DEADLINE = 630.0
_EPS = 1e-6
_SQ3 = np.sqrt(3.0)
_SQ11 = np.sqrt(11.0)


def _hex_verts(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _face_normals(a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.column_stack([np.cos(th), np.sin(th)])


def _pair_gap(va, vb, na, nb):
    best = -np.inf
    for n in (na, nb):
        pa = va @ n
        pb = vb @ n
        g = max(pb.min() - pa.max(), pa.min() - pb.max())
        if g > best:
            best = g
    return best


def _outer_side(shifted, a_deg):
    n = _face_normals(a_deg)
    dots = shifted @ n.T
    r = dots.max()
    return 2.0 * r / _SQ3


def _best_orientation(shifted, n_ang=181):
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0.0, 60.0, n_ang):
        s = _outer_side(shifted, a_deg)
        if s < best_s:
            best_s, best_a = s, a_deg
    return best_s, best_a


def _objective(params):
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_verts(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            g = _pair_gap(verts[i], verts[j], norms[i], norms[j])
            if g < min_gap:
                min_gap = g
    allv = verts.reshape(-1, 2)
    cent = allv.mean(axis=0)
    shifted = allv - cent
    side, _ = _best_orientation(shifted, n_ang=25)
    return side, min_gap


def _objective_scalar(params):
    side, min_gap = _objective(params)
    if min_gap < _EPS:
        return 1e6 * (_EPS - min_gap) + side
    return side


def _valid(params):
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_verts(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
    for i in range(11):
        for j in range(i + 1, 11):
            if _pair_gap(verts[i], verts[j], norms[i], norms[j]) <= _EPS:
                return None
    allv = verts.reshape(-1, 2)
    cent = allv.mean(axis=0)
    shifted = allv - cent
    side, a_deg = _best_orientation(shifted, n_ang=181)
    if side < _SQ11 - 1e-9:
        return None
    ov = _hex_verts(0.0, 0.0, side, a_deg)
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        e = p2 - p1
        c = np.cross(e, shifted - p1)
        if c.min() < -1e-6:
            return None
    return side, (cent[0], cent[1], a_deg)


def _flower_seed():
    p = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4, 5]:
        ang = np.deg2rad(60.0 * k + 30.0)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    return np.array(p).flatten()


def _ring_seed():
    p = []
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    return np.array(p).flatten()


def _band_seed():
    h = _SQ3 * 2.0
    rows = [(0.0, 3), (h, 2), (2 * h, 3), (3 * h, 2), (4 * h, 3)]
    p = []
    for y, cnt in rows:
        for i in range(cnt):
            p.append([(i - (cnt - 1) / 2.0) * 2.0, y, 0.0])
    return np.array(p).flatten()


def _hex_grid_seed():
    p = []
    for row in range(3):
        y = row * _SQ3 * 2.0
        offset = (row % 2) * 1.0
        for col in range(4):
            x = col * 2.0 + offset - 3.0
            p.append([x, y, 0.0])
    return np.array(p).flatten()


def _cluster_seed():
    p = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4, 6]:
        ang = np.deg2rad(30.0 + 60.0 * k)
        p.append([3.464 * np.cos(ang), 3.464 * np.sin(ang), 0.0])
    return np.array(p).flatten()


def _hex_grid_seed2():
    # Tighter hex grid with different offset
    p = []
    for row in range(3):
        y = row * _SQ3 * 2.0
        offset = (row % 2) * 1.0
        for col in range(4):
            x = col * 2.0 + offset - 1.5
            p.append([x, y, 0.0])
    return np.array(p).flatten()


def _contact_constraint_solve(params_init, contacts):
    """
    Solve for positions such that specified pairs of hexagons are in contact.
    contacts: list of (i, j, normal_index) where normal_index is 0-5
    """
    n = 11
    def equations(params):
        eqs = []
        verts = np.empty((11, 6, 2))
        norms = np.empty((11, 6, 2))
        for i in range(11):
            cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
            verts[i] = _hex_verts(cx, cy, 1.0, ad)
            norms[i] = _face_normals(ad)
        
        for (i, j, ni) in contacts:
            n_vec = norms[i][ni]
            # Project both hexagons onto this normal
            pa = verts[i] @ n_vec
            pb = verts[j] @ n_vec
            # Gap: max of (min_b - max_a, min_a - max_b)
            gap = max(pb.min() - pa.max(), pa.min() - pb.max())
            eqs.append(gap)  # We want gap = 0 (contact)
        
        return eqs
    
    try:
        sol, info, ier, msg = fsolve(equations, params_init, full_output=True, maxfev=5000)
        if ier == 1:
            return sol
    except:
        pass
    return None


def hexagon_packing_11():
    seeds = [_flower_seed(), _ring_seed(), _band_seed(), _hex_grid_seed(), _cluster_seed(), _hex_grid_seed2()]

    best = None
    best_side = np.inf

    def consider(params):
        nonlocal best, best_side
        v = _valid(params)
        if v is None:
            return
        side, outer = v
        if side < best_side:
            best_side = side
            inner = np.empty((11, 3))
            for i in range(11):
                inner[i] = [params[3 * i] - outer[0],
                            params[3 * i + 1] - outer[1],
                            params[3 * i + 2]]
            best = (side, inner, np.array(outer))

    # First pass: Nelder-Mead from each seed
    for seed in seeds:
        if time.time() > _DEADLINE - 90.0:
            break
        res = minimize(_objective_scalar, seed, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8,
                                'fatol': 1e-8, 'adaptive': True})
        consider(res.x)

    # Second pass: contact-constrained refinement around best solution
    if best is not None and time.time() < _DEADLINE - 120.0:
        best_side_cur, inner_cur, outer_cur = best
        params_best = np.empty(33)
        for i in range(11):
            params_best[3 * i] = inner_cur[i, 0] + outer_cur[0]
            params_best[3 * i + 1] = inner_cur[i, 1] + outer_cur[1]
            params_best[3 * i + 2] = inner_cur[i, 2]

        # Try contact-constrained solving with various contact patterns
        # For a hexagonal packing, typical contacts are:
        # Center (0) with ring (1-6), and ring with outer (7-10)
        contact_patterns = [
            [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4), (0, 6, 5)],
            [(1, 7, 0), (2, 8, 1), (3, 9, 2), (4, 10, 3)],
            [(0, 1, 0), (1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 5, 4), (5, 6, 5)],
        ]
        
        for contacts in contact_patterns:
            if time.time() > _DEADLINE - 90.0:
                break
            sol = _contact_constraint_solve(params_best, contacts)
            if sol is not None:
                consider(sol)

    # Third pass: local refinement around best solution
    if best is not None and time.time() < _DEADLINE - 60.0:
        best_side_cur, inner_cur, outer_cur = best
        params_best = np.empty(33)
        for i in range(11):
            params_best[3 * i] = inner_cur[i, 0] + outer_cur[0]
            params_best[3 * i + 1] = inner_cur[i, 1] + outer_cur[1]
            params_best[3 * i + 2] = inner_cur[i, 2]

        rng = np.random.RandomState(42)
        for attempt in range(15):
            if time.time() > _DEADLINE - 60.0:
                break
            perturb = rng.normal(0, 0.1, 33)
            perturbed = params_best + perturb
            res = minimize(_objective_scalar, perturbed, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-9,
                                    'fatol': 1e-9, 'adaptive': True})
            consider(res.x)

    if best is None:
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ], dtype=float)
        return inner, np.array([0.0, 0.0, 0.0]), 8.0

    side, inner, outer = best
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
