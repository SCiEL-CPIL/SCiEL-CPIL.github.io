"""Convex-polygon parameterization for Heilbronn n=14."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
TOTAL_TURN = (N - 2) * np.pi  # 12*pi

# ---------- Core geometry ----------

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _all_triangle_areas(pts):
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    out = []
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n - 1):
            v1x = x[j] - xi
            v1y = y[j] - yi
            v2x = x[j+1:] - xi
            v2y = y[j+1:] - yi
            cross = v1x * v2y - v1y * v2x
            out.append(0.5 * np.abs(cross))
    return np.concatenate(out) if out else np.array([0.0])

def _min_triangle_area(pts):
    return float(np.min(_all_triangle_areas(pts)))

def _score(pts):
    h = _hull_area(pts)
    if h < 1e-10:
        return 0.0
    m = _min_triangle_area(pts)
    return m / h

def _soft_min(areas, beta):
    """Smooth approximation to min(areas) via -logsumexp(-beta*areas)/beta."""
    z = -beta * areas
    mz = np.max(z)
    if mz < -500:
        lse = mz
    else:
        lse = mz + np.log(np.sum(np.exp(z - mz)))
    return -lse / beta

def _neg_score(pts, beta=200.0):
    """Objective to minimize: -soft_min(areas)/hull_area."""
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if areas.size == 0:
        return 1e10
    if np.min(areas) < 1e-12:
        return 1e10
    sm = _soft_min(areas, beta)
    if sm < 1e-12:
        return 1e10
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e10
    return -sm / h

# ---------- Convex polygon parameterization ----------
# params: [log_e1..log_e14 (14), log_t1..log_t13 (13)]  -> 27 params
# turn_i = exp(log_t_i), i=1..13 ; turn_14 = TOTAL_TURN - sum(turn_1..13)
# edge_i = exp(log_e_i)
# Build polygon by walking; must close (turns sum to 12*pi guarantees this for a
# closed convex polygon when edge lengths are consistent). We build by placing
# vertices via cumulative angles; closure is enforced by the turn-sum constraint
# combined with the fact that a closed convex polygon's edge vectors sum to zero.
# In practice, we build the polygon from edges+turns and then translate to centroid.
# To guarantee closure, we use the standard construction: given edge lengths e_i
# and exterior angles (turns) theta_i summing to 2*pi (here we use interior sum
# (N-2)pi -> exterior turns sum 2pi). We use EXTERIOR turns.

def _params_to_points(params):
    """27 params -> 14 points (convex polygon, closed)."""
    le = params[0:14]
    lt = params[14:27]
    e = np.exp(le)
    # exterior turns: 14 turns summing to 2*pi
    t = np.exp(lt)  # 13 turns
    total13 = t.sum()
    t14 = 2.0 * np.pi - total13
    if t14 <= 1e-6:
        # infeasible; clamp by rescaling
        t = t * (2.0 * np.pi * 0.95) / total13
        t14 = 2.0 * np.pi - t.sum()
        if t14 <= 1e-6:
            return None
    turns = np.concatenate([t, [t14]])

    # Build edge vectors
    angles = np.zeros(N)
    angles[0] = 0.0
    for i in range(1, N):
        angles[i] = angles[i-1] + turns[i-1]
    ex = e * np.cos(angles)
    ey = e * np.sin(angles)

    # Cumulative to get vertices (start at origin)
    vx = np.concatenate([[0.0], np.cumsum(ex)])[:N]
    vy = np.concatenate([[0.0], np.cumsum(ey)])[:N]
    pts = np.column_stack([vx, vy])

    # The polygon may not close exactly due to numeric; check closure
    # (turns sum to 2pi and edges chosen => should close). Translate to centroid.
    cx, cy = pts.mean(axis=0)
    pts = pts - np.array([cx, cy])
    return pts

def _obj_convex(params, beta=200.0):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _params_to_points(params)
    if pts is None or np.any(~np.isfinite(pts)):
        return 1e10
    # Ensure it's a valid convex hull with positive area
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e10
    return _neg_score(pts, beta)

# ---------- Initializers ----------

def _regular_polygon_params(perturb=0.0, rng=None):
    """Regular 14-gon: equal edges, equal exterior turns = 2pi/14."""
    e = np.ones(14)
    turns = np.full(13, 2.0 * np.pi / 14.0)
    le = np.log(e)
    lt = np.log(turns)
    params = np.concatenate([le, lt])
    if perturb > 0 and rng is not None:
        params = params + rng.normal(0, perturb, size=params.shape)
    return params

def _diverse_initializers():
    inits = []
    base = _regular_polygon_params()
    inits.append(base.copy())
    rng = np.random.default_rng(0)
    for seed in range(20):
        rng = np.random.default_rng(seed * 101 + 7)
        p = _regular_polygon_params(perturb=0.15, rng=rng)
        inits.append(p)
    # A few with larger perturbation
    for seed in range(5):
        rng = np.random.default_rng(seed * 37 + 13)
        p = _regular_polygon_params(perturb=0.4, rng=rng)
        inits.append(p)
    return inits

# ---------- Optimizers ----------

def _slsqp(x0, deadline, beta=200.0, max_iter=800):
    if time.time() > deadline - 5.0:
        return None
    try:
        return minimize(_obj_convex, x0, method='SLSQP',
                        args=(beta,),
                        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
    except Exception:
        return None

def _nm_free(pts, deadline, max_iter=1500):
    """Free-point Nelder-Mead polish."""
    if time.time() > deadline - 5.0:
        return None
    x0 = pts.flatten()
    def obj(p):
        return _neg_score(p.reshape(N, 2), beta=150.0)
    try:
        return minimize(obj, x0, method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-10,
                                 'fatol': 1e-12, 'adaptive': True})
    except Exception:
        return None

def _nm_convex(params, deadline, max_iter=1500, beta=200.0):
    if time.time() > deadline - 5.0:
        return None
    try:
        return minimize(_obj_convex, params, method='Nelder-Mead',
                        args=(beta,),
                        options={'maxiter': max_iter, 'xatol': 1e-10,
                                 'fatol': 1e-12, 'adaptive': True})
    except Exception:
        return None

# ---------- Main ----------

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        h = _hull_area(pts)
        if h < 1e-10:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    def _consider_params(params):
        pts = _params_to_points(params)
        if pts is not None:
            _consider(pts)

    # Phase 1: Multi-start SLSQP on convex-polygon params
    inits = _diverse_initializers()
    for x0 in inits:
        if time.time() > deadline - 50:
            break
        r = _slsqp(x0, deadline, beta=200.0, max_iter=600)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider_params(r.x)

    # Phase 2: Nelder-Mead refinement on best convex params (from best points,
    # we don't have params directly; instead re-run NM from each init that gave
    # good score). We track best params.
    # Re-evaluate: find best params by re-optimizing the top few.
    # Simpler: run NM from the regular polygon and a couple of perturbations.
    for x0 in inits[:5]:
        if time.time() > deadline - 30:
            break
        r = _nm_convex(x0, deadline, max_iter=1200, beta=200.0)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider_params(r.x)

    # Phase 3: Free-point polish from best convex-polygon solution
    if best_pts is not None and time.time() < deadline - 15:
        r = _nm_free(best_pts, deadline, max_iter=1200)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider(r.x.reshape(N, 2))

    # Phase 4: Second free-point polish with different beta
    if best_pts is not None and time.time() < deadline - 8:
        def obj2(p):
            return _neg_score(p.reshape(N, 2), beta=80.0)
        x0 = best_pts.flatten()
        try:
            r = minimize(obj2, x0, method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-10,
                                  'fatol': 1e-12, 'adaptive': True})
            if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
                _consider(r.x.reshape(N, 2))
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Normalize to unit area
    h = _hull_area(best_pts)
    if h > 1e-10:
        final_pts = best_pts * np.sqrt(1.0 / h)
    else:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        final_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Safety
    try:
        if _hull_area(final_pts) < 1e-10:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            final_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    except Exception:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        final_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return final_pts
