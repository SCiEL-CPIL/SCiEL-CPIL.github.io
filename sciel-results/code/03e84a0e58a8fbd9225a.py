import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all triples (i<j<k)
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()
N_TRI = IDX_I.shape[0]

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return float(np.min(_all_areas(points)))

def _crosses(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

def _compute_hull_area_fast(points):
    # Andrew monotone chain, O(n log n); returns signed-area magnitude
    p = np.ascontiguousarray(points, dtype=float)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    n = p.shape[0]
    if n < 3:
        return 0.0
    def half(pts):
        stack = []
        for pt in pts:
            while len(stack) >= 2 and _crosses(stack[-2], stack[-1], pt) <= 0.0:
                stack.pop()
            stack.append(pt)
        return stack
    lower = half(p)
    upper = half(p[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)

def _compute_hull_area(points):
    return _compute_hull_area_fast(points)

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(params):
    angles = params[:N_HALF]
    radii = params[N_HALF:]
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective(params):
    points = _params_to_points(params)
    return -_score(points)

def _objective_soft(params, tau=0.02):
    """Smooth surrogate: -(softmin triangle area)/(hull area), softmin via log-sum-exp."""
    points = _params_to_points(params)
    areas = _all_areas(points)
    amax = areas.max()
    if amax < 1e-12:
        return 0.0
    softmin = amax * np.exp(-np.sum(np.exp((areas - amax) / tau)) / N_TRI)
    hull_area = _compute_hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    return -(softmin / hull_area)

def _bounds_sym():
    bounds = []
    for _ in range(N_HALF):
        bounds.append((0.02, np.pi - 0.02))
    for _ in range(N_HALF):
        bounds.append((0.05, 3.0))
    return bounds

def _nm(params, t_left):
    t0 = time.time()
    try:
        res = minimize(_objective, params, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13,
                                'adaptive': True})
        if np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return params, 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill
    bounds = _bounds_sym()

    best_score = 0.0
    best_params = None

    # --- Build diverse initial parameter sets ---
    inits = []
    rng = np.random.default_rng(42)
    base_angles = np.linspace(0.1, np.pi - 0.1, N_HALF)
    for r_scale in [0.5, 0.7, 1.0, 1.3]:
        inits.append(np.concatenate([base_angles, np.full(N_HALF, r_scale)]))
    for _ in range(15):
        a = np.sort(rng.uniform(0.1, np.pi - 0.1, N_HALF))
        r = rng.uniform(0.5, 1.5, N_HALF)
        inits.append(np.concatenate([a, r]))
    for _ in range(15):
        a = np.sort(rng.uniform(0.05, np.pi - 0.05, N_HALF))
        r = rng.uniform(0.2, 2.5, N_HALF)
        inits.append(np.concatenate([a, r]))

    def _update(p, s):
        nonlocal best_score, best_params
        if s > best_score:
            best_score = s
            best_params = p.copy()

    # --- Phase 1: Global exploration (DE) ---
    if time.time() < deadline - 70:
        try:
            de = differential_evolution(_objective, bounds, seed=42, maxiter=40,
                                        popsize=12, tol=1e-5, polish=False,
                                        init='sobol', mutation=(0.3, 1.0),
                                        recombination=0.7)
            if np.isfinite(de.fun):
                _update(de.x, -de.fun)
        except Exception:
            pass

    # --- Phase 2: Score inits, keep top seeds ---
    scored = []
    for init in inits:
        if time.time() > deadline - 45:
            break
        try:
            s = -_objective(init)
            scored.append((s, init))
        except Exception:
            pass
    scored.sort(key=lambda x: x[0], reverse=True)

    # --- Phase 3: Smooth-surrogate basin finding + nonsmooth refine ---
    seeds = []
    if best_params is not None:
        seeds.append(best_params)
    for s, p in scored[:6]:
        seeds.append(p)
    seeds.append(np.concatenate([np.linspace(0.1, np.pi - 0.1, N_HALF), np.ones(N_HALF)]))

    for seed in seeds:
        if time.time() > deadline - 12:
            break
        # 3a. Navigate the nonsmooth landscape with a smooth surrogate
        p_soft = seed
        for tau in [0.05, 0.02, 0.008]:
            if time.time() > deadline - 10:
                break
            try:
                r = minimize(lambda pp: _objective_soft(pp, tau), p_soft,
                             method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10,
                                      'fatol': 1e-12, 'adaptive': True})
                if np.isfinite(r.fun):
                    p_soft = r.x
            except Exception:
                pass
        # 3b. Nonsmooth refine from the soft basin
        p_ref, s_ref = _nm(p_soft, deadline - time.time())
        _update(p_ref, s_ref)
        # 3c. Also refine the raw seed (in case soft-min drifted)
        p_ref2, s_ref2 = _nm(seed, deadline - time.time())
        _update(p_ref2, s_ref2)

    # --- Phase 4: Coordinate-wise + high-precision polish on the best ---
    if best_params is not None and time.time() < deadline - 4:
        p = best_params
        # Powell coordinate-wise
        try:
            r = minimize(_objective, p, method='Powell',
                         options={'maxiter': 800, 'xtol': 1e-12, 'ftol': 1e-14})
            if np.isfinite(r.fun):
                _update(r.x, -r.fun)
        except Exception:
            pass
        # Final tight Nelder-Mead
        if time.time() < deadline - 2:
            p = best_params
            for it in range(2):
                if time.time() > deadline - 1:
                    break
                try:
                    r = minimize(_objective, p, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-13,
                                          'fatol': 1e-15, 'adaptive': True})
                    if np.isfinite(r.fun):
                        s = -r.fun
                        if s > best_score:
                            best_score = s
                            p = r.x
                            best_params = r.x.copy()
                except Exception:
                    break

    # --- Fallback ---
    if best_params is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        best_points = _params_to_points(best_params)

    # Normalize to unit area
    hull_area = _compute_hull_area(best_points)
    if hull_area > 1e-12:
        best_points = best_points * np.sqrt(1.0 / hull_area)
    return best_points
