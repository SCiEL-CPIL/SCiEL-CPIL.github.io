import time
import math
import numpy as np

N = 12

# --- geometry helpers ---------------------------------------------------------
# Unit regular hexagon, flat-top, centered at origin:
# vertices at angles 0,60,...,300 (circumradius 1)
_UNIT_HEX = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# 3 support directions (outward normals of the 3 edge pairs) for a flat-top hexagon
_AXES = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3) / 2.0],
    [-0.5, math.sqrt(3) / 2.0],
])

def _hex_pts(x, y, ang_deg):
    """Vertices of unit hexagon centered at (x,y) rotated by ang_deg."""
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return _UNIT_HEX @ R.T + np.array([x, y])

def _support_of_hex(center, ang_deg, axes):
    """Max projection of a unit hexagon (center, rot) onto given axes (k x 2). Returns (k,)."""
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    # vertices in local frame
    v = _UNIT_HEX @ R.T  # (6,2)
    proj = v @ axes.T     # (6, k)
    return np.max(proj, axis=0) + center @ axes.T  # (k,)

def _outer_support_limit(S, axes):
    """Max allowed support for outer hexagon of side S (same orientation as axes)."""
    return S * math.sqrt(3) / 2.0

def _feasibility(params, S):
    """params: list of (x,y,ang). Returns (ok, violation).
    violation = max over pairs of (2 - dist) and over hex of (support - limit)."""
    n = len(params)
    pts = [np.array([p[0], p[1]]) for p in params]
    viol = 0.0
    # pairwise separation
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            v = 2.0 - d
            if v > viol:
                viol = v
    # containment
    lim = _outer_support_limit(S, _AXES)
    for i in range(n):
        sup = _support_of_hex(pts[i], params[i][2], _AXES)
        v = float(np.max(sup - lim))
        if v > viol:
            viol = v
    return viol <= 1e-9, viol

def _objective(params, S):
    """Penalized objective: S + penalty * (violation)^2. Lower better."""
    _, viol = _feasibility(params, S)
    return S + 10.0 * viol * viol

# --- initial placement: hexagonal cluster of 12 ------------------------------
def _initial_params():
    """7-hex cluster + 5 more around it, all axis-aligned (ang=0)."""
    # axial coords (q,r) -> cartesian for flat-top hex grid with size 1
    # neighbor spacing: for flat-top, centers of adjacent unit hexes are 2*inradius = sqrt(3) apart
    sp = math.sqrt(3)  # distance between adjacent unit hexagon centers
    def ax(q, r):
        x = sp * (q + 0.5 * r)
        y = sp * (math.sqrt(3) / 2.0 * r)
        return (x, y, 0.0)
    # 7-hex cluster: (0,0) and its 6 neighbors
    cluster = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
    # 5 more around the cluster (a ring segment)
    extra = [(2,0),(-2,0),(0,2),(0,-2),(2,-2)]
    coords = cluster + extra
    return [ax(q, r) for (q, r) in coords]

# --- local optimization -------------------------------------------------------
def _optimize(params0, S0, deadline):
    """Minimize _objective over params (3N) and S (1). Returns best (params, S, obj)."""
    n = len(params0)
    dim = 3 * n + 1
    x0 = []
    for (x, y, a) in params0:
        x0 += [x, y, a]
    x0.append(S0)
    x = np.array(x0, dtype=float)
    best = (x.copy(), _objective(_unpack(x), x[-1]))

    def _unpack(xx):
        ps = []
        for i in range(n):
            ps.append((xx[3*i], xx[3*i+1], xx[3*i+2]))
        return ps

    def _obj(xx):
        return _objective(_unpack(xx), xx[-1])

    # Nelder-Mead (scipy) with time guard
    t_end = deadline
    try:
        from scipy.optimize import minimize
        def f(xx):
            if time.time() > t_end - 2:
                return _obj(xx) + 1e6
            return _obj(xx)
        res = minimize(f, x0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        if _obj(res.x) < best[1]:
            best = (res.x.copy(), _obj(res.x))
    except Exception:
        pass

    # BFGS polish
    if time.time() < t_end - 5:
        try:
            from scipy.optimize import minimize
            def f2(xx):
                if time.time() > t_end - 2:
                    return _obj(xx) + 1e6
                return _obj(xx)
            res2 = minimize(f2, best[0], method='BFGS',
                            options={'maxiter': 2000, 'gtol': 1e-5})
            if _obj(res2.x) < best[1]:
                best = (res2.x.copy(), _obj(res2.x))
        except Exception:
            pass
    return best

# --- main construct -----------------------------------------------------------
def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s

    params0 = _initial_params()
    # initial S: compute a feasible outer side for the initial layout
    # find S such that containment holds (pairwise already holds by construction)
    # compute required S from supports
    def _required_S(params):
        need = 0.0
        for (x, y, a) in params:
            sup = _support_of_hex(np.array([x, y]), a, _AXES)
            need = max(need, float(np.max(sup)))
        return need / (math.sqrt(3) / 2.0) + 1e-9
    S0 = _required_S(params0)

    # Try a few randomized starts if time allows
    best_params, best_S, best_obj = None, None, None
    starts = [params0]
    rng = np.random.default_rng(12345)
    # a couple of jittered starts
    for _ in range(3):
        jp = [(x + rng.normal(0, 0.1), y + rng.normal(0, 0.1), float(rng.normal(0, 5.0))) for (x, y, a) in params0]
        starts.append(jp)

    for st in starts:
        if time.time() > deadline - 10:
            break
        # ensure initial S feasible for this start
        Ss = _required_S(st)
        # but also ensure pairwise: if not, increase S won't help pairwise; check
        ok, viol = _feasibility(st, Ss)
        if not ok:
            # pairwise violation; just proceed, objective penalizes
            pass
        p, o = _optimize(st, Ss, deadline)
        if best_obj is None or o < best_obj:
            best_obj = o
            best_params = _unpack(p)
            best_S = p[-1]

    if best_params is None:
        best_params = params0
        best_S = S0

    # Final: if the best is infeasible, compute a safe S that guarantees containment
    # (pairwise is fixed by layout; we can always enlarge S to contain)
    ok, viol = _feasibility(best_params, best_S)
    if not ok:
        # enlarge S to fix containment; pairwise may still be violated -> return best anyway
        Ss = _required_S(best_params)
        # check pairwise at this S
        ok2, viol2 = _feasibility(best_params, Ss)
        if ok2:
            best_S = Ss
        else:
            # pairwise violated; try to keep best_S but ensure containment
            if Ss > best_S:
                best_S = Ss

    inner = [[p[0], p[1], p[2]] for p in best_params]
    return inner, (0.0, 0.0), float(best_S), 0.0
