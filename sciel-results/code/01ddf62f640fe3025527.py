"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

# ---------- Core geometry ----------

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n - 1):
            v1x = x[j] - xi
            v1y = y[j] - yi
            v2x = x[j+1:] - xi
            v2y = y[j+1:] - yi
            cross = v1x * v2y - v1y * v2x
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area

def _all_areas(points):
    """All triangle areas."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
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

def _hull_area(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0

def _score(points):
    h = _hull_area(points)
    if h < 1e-10:
        return 0.0
    m = _min_triangle_area(points)
    return m / h

def _normalize_unit_area(points):
    try:
        h = _hull_area(points)
        if h > 1e-10:
            return points * np.sqrt(1.0 / h)
    except Exception:
        pass
    xs, ys = points[:, 0], points[:, 1]
    w, hh = xs.max() - xs.min(), ys.max() - ys.min()
    if w > 1e-10 and hh > 1e-10:
        return points / np.sqrt(w * hh)
    return points

# ---------- Hexagonal lattice parameterization ----------
# 14 points = 7 pairs (a_i, -a_i) where a_i = b1*u_i + b2*v_i
# b1, b2 are lattice basis vectors (4 params)
# u_i, v_i are integer-ish coordinates (14 params)
# Total: 18 params

def _lattice_params_to_points(params):
    """params: [b1x, b1y, b2x, b2y, u1..u7, v1..v7] -> 14 points"""
    b1 = np.array([params[0], params[1]])
    b2 = np.array([params[2], params[3]])
    u = params[4:11]
    v = params[11:18]
    half = np.array([u[i]*b1 + v[i]*b2 for i in range(7)])
    return np.vstack([half, -half])

def _objective_lattice(params):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _lattice_params_to_points(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    m = _min_triangle_area(pts)
    if m < 1e-12:
        return 1e10
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e10
    return h / m

def _objective_lattice_soft(params, beta=150.0):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _lattice_params_to_points(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = np.max(z)
    if mz < -500:
        lse = mz
    else:
        lse = mz + np.log(np.sum(np.exp(z - mz)))
    sm = -lse / beta
    if sm < 1e-12:
        return 1e10
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e10
    return h / sm

# ---------- Hexagonal lattice initializers ----------
def _hex_initializers():
    """Generate diverse lattice-based initial parameters."""
    inits = []
    # Standard hexagonal lattice: b1=(1,0), b2=(0.5, sqrt(3)/2)
    # 7 points in a hexagonal cluster pattern
    s3 = np.sqrt(3)
    
    # Pattern 1: hexagonal ring + center-ish
    # Use integer coords in hex basis
    patterns = [
        # (u, v) pairs - 7 points
        [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1)],
        [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (1,2)],
        [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,2)],
        [(1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2)],
        [(0,0), (2,0), (0,2), (2,2), (1,0), (0,1), (1,1)],
        [(1,1), (2,0), (0,2), (2,2), (1,0), (0,1), (3,1)],
        [(0,0), (1,0), (0,1), (1,1), (2,1), (1,2), (2,2)],
        [(1,0), (1,1), (2,0), (2,1), (0,1), (0,2), (1,2)],
    ]
    
    for pattern in patterns:
        u = np.array([p[0] for p in pattern], dtype=float)
        v = np.array([p[1] for p in pattern], dtype=float)
        # Hexagonal basis
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.5, s3/2.0])
        params = np.concatenate([[b1[0], b1[1], b2[0], b2[1]], u, v])
        inits.append(params)
    
    # Slightly rotated hexagonal lattices
    for rot in [0.1, -0.1, 0.2, -0.2, 0.3]:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        b1 = R @ np.array([1.0, 0.0])
        b2 = R @ np.array([0.5, s3/2.0])
        pattern = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (1,2)]
        u = np.array([p[0] for p in pattern], dtype=float)
        v = np.array([p[1] for p in pattern], dtype=float)
        params = np.concatenate([[b1[0], b1[1], b2[0], b2[1]], u, v])
        inits.append(params)
    
    # Random perturbations of hexagonal
    rng = np.random.default_rng(7)
    for seed in range(15):
        rng = np.random.default_rng(seed * 13 + 3)
        # Random basis close to hexagonal
        b1 = np.array([1.0, 0.0]) + rng.normal(0, 0.15, 2)
        b2 = np.array([0.5, s3/2.0]) + rng.normal(0, 0.15, 2)
        pattern = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (1,2)]
        u = np.array([p[0] for p in pattern], dtype=float) + rng.normal(0, 0.1, 7)
        v = np.array([p[1] for p in pattern], dtype=float) + rng.normal(0, 0.1, 7)
        params = np.concatenate([[b1[0], b1[1], b2[0], b2[1]], u, v])
        inits.append(params)
    
    return inits

# ---------- Free point parameterization (fallback / comparison) ----------
def _free_params_to_points(params):
    """14 points, 28 params."""
    return params.reshape(14, 2)

def _objective_free(params):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _free_params_to_points(params)
    m = _min_triangle_area(pts)
    if m < 1e-12:
        return 1e10
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e10
    return h / m

# ---------- Optimizers ----------
def _nm(x0, deadline, max_iter=2000, obj=None):
    if obj is None: obj = _objective_lattice
    if time.time() > deadline - 2.0:
        return None
    try:
        return minimize(obj, x0, method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=None):
    if obj is None: obj = _objective_lattice_soft
    if time.time() > deadline - 5.0:
        return None
    try:
        return minimize(obj, x0, method='L-BFGS-B',
                        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
    except Exception:
        return None

def _powell(x0, deadline, max_iter=2000, obj=None):
    if obj is None: obj = _objective_lattice
    if time.time() > deadline - 3.0:
        return None
    try:
        return minimize(obj, x0, method='Powell',
                        options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-14})
    except Exception:
        return None

def _de(obj, bounds, deadline, max_iter=100, popsize=15, seed=42):
    if time.time() > deadline - 15.0:
        return None
    try:
        return differential_evolution(obj, bounds, maxiter=max_iter,
                                      popsize=popsize, tol=1e-10, atol=1e-12,
                                      seed=seed, polish=True, init='latinhypercube')
    except Exception:
        return None

# ---------- Main ----------
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    def _consider_from_lattice(params):
        pts = _lattice_params_to_points(params)
        _consider(pts)

    # Phase 1: Differential evolution on lattice params
    # bounds: b1x, b1y, b2x, b2y in [-2, 2]; u,v in [-3, 3]
    bounds = [(-2.0, 2.0)] * 4 + [(-3.0, 3.0)] * 14
    res = _de(_objective_lattice, bounds, deadline, max_iter=100, popsize=15, seed=42)
    if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
        _consider_from_lattice(res.x)

    # Phase 2: Local refinement from diverse lattice starts
    inits = _hex_initializers()
    if res is not None:
        inits.insert(0, res.x)

    for x0 in inits:
        if time.time() > deadline - 45:
            break
        r = _nm(x0, deadline, max_iter=1500, obj=_objective_lattice)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider_from_lattice(r.x)

    # Phase 3: Gradient polish on best lattice solution
    # Find best params from inits
    best_params = None
    best_obj = 1e10
    for x0 in inits:
        if time.time() > deadline - 30:
            break
        obj_val = _objective_lattice(x0)
        if obj_val < best_obj:
            best_obj = obj_val
            best_params = x0.copy()

    if best_params is not None and time.time() < deadline - 25:
        r = _lbfgs(best_params, deadline, max_iter=3000, obj=_objective_lattice_soft)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider_from_lattice(r.x)
            best_params = r.x.copy()

    # Phase 4: Powell polish
    if best_params is not None and time.time() < deadline - 15:
        r = _powell(best_params, deadline, max_iter=2000, obj=_objective_lattice)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider_from_lattice(r.x)
            best_params = r.x.copy()

    # Phase 5: Final NM with tight tolerance
    if best_params is not None and time.time() < deadline - 8:
        r = _nm(best_params, deadline, max_iter=1000, obj=_objective_lattice)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider_from_lattice(r.x)

    # Phase 6: Also try free-point optimization from the best lattice solution
    if best_pts is not None and time.time() < deadline - 10:
        free_params = best_pts.flatten()
        r = _nm(free_params, deadline, max_iter=800, obj=_objective_free)
        if r is not None and np.isfinite(r.fun) and r.fun < 1e9:
            _consider(_free_params_to_points(r.x))

    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        h = _hull_area(final_points)
        if h < 1e-10:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
