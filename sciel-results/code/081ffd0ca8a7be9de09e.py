import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 145.0  # wall-clock guard (evaluator kills at 150s)

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_ring_config(n, frac=0.5, phase=0.5):
    n_outer = n - 7
    n_inner = 7
    outer = _regular_polygon(n_outer, 1.0, 0.0)
    inner = _regular_polygon(n_inner, frac, phase * 2 * np.pi / n_inner)
    return np.vstack([outer, inner])

def _dumbbell_config(pair_centers, v):
    """
    Construct 14 points from 7 pair-centers and a displacement vector v.
    Each pair is at center ± v/2.
    """
    pts = []
    for c in pair_centers:
        pts.append(c + 0.5 * v)
        pts.append(c - 0.5 * v)
    return np.array(pts)

def _score_dumbbell(flat):
    """
    flat: [7*2 (pair centers) + 2 (v vector)] = 16 values
    """
    pc = flat[:14].reshape(7, 2)
    v = flat[14:16]
    pts = _dumbbell_config(pc, v)
    return _score(pts)

def _neg_score_dumbbell(flat):
    return -_score_dumbbell(flat)

def _normalize_dumbbell(flat):
    """
    Normalize the dumbbell configuration so hull area = 1.
    Returns normalized flat representation.
    """
    pc = flat[:14].reshape(7, 2)
    v = flat[14:16]
    pts = _dumbbell_config(pc, v)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return flat
    scale = 1.0 / np.sqrt(ha)
    pc_norm = pc * scale
    v_norm = v * scale
    return np.concatenate([pc_norm.flatten(), v_norm])

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3]:
        starts.append(_regular_polygon(N, 1.0, rot))
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.45, 0.5, 0.55, 0.6, 0.65]:
        for phase in [0.0, 0.5, 0.25, 0.75]:
            starts.append(_two_ring_config(N, frac, phase))
    return starts

def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)

    # Phase 1: Evaluate structured starts
    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if left() <= 2.0:
            break
        sn = _normalize(s)
        scored.append((_score(sn), sn))
    scored.sort(key=lambda x: -x[0])

    best_score, best_pts = 0.0, _normalize(_regular_polygon(N, 1.0, 0.0))
    if scored:
        s0, p0 = scored[0]
        best_score, best_pts = s0, p0.copy()

    def try_opt(x0, method, maxiter, fatol, xatol):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'adaptive': True, 'maxfev': 25000}
            res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Phase 2: Polish best starts with local optimization
    for sc, pts in scored:
        if left() <= 6.0:
            break
        try_opt(pts, 'Nelder-Mead', 20000, 1e-13, 1e-10)
        if left() <= 10.0:
            break
        try_opt(pts, 'Powell', 6000, 1e-13, 1e-11)

    # Phase 3: Dumbbell parameterization with differential evolution
    if left() > 20.0:
        # Build initial dumbbell starts from best point configurations
        dumbbell_starts = []
        
        # Start 1: Regular heptagon pair centers with small v
        hept = _regular_polygon(7, 1.0, 0.0)
        for v_mag in [0.05, 0.1, 0.15, 0.2]:
            for v_angle in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
                v = v_mag * np.array([np.cos(v_angle), np.sin(v_angle)])
                flat = np.concatenate([hept.flatten(), v])
                dumbbell_starts.append(flat)
        
        # Start 2: From best point configuration - try to extract pair structure
        # Sort points by angle from centroid and pair adjacent
        centroid = best_pts.mean(axis=0)
        angles = np.arctan2(best_pts[:,1] - centroid[1], best_pts[:,0] - centroid[0])
        order = np.argsort(angles)
        sorted_pts = best_pts[order]
        
        # Pair adjacent points (0-1, 2-3, ..., 12-13)
        pair_centers = []
        for i in range(0, 14, 2):
            pair_centers.append(0.5 * (sorted_pts[i] + sorted_pts[i+1]))
        pair_centers = np.array(pair_centers)
        
        # Estimate v from first few pairs
        v_estimates = []
        for i in range(0, 14, 2):
            v_estimates.append(sorted_pts[i] - sorted_pts[i+1])
        v_est = np.mean(v_estimates, axis=0)
        
        flat = np.concatenate([pair_centers.flatten(), v_est])
        dumbbell_starts.append(flat)
        
        # Try multiple v magnitudes/angles around the estimated v
        v_mag = np.linalg.norm(v_est)
        v_angle = np.arctan2(v_est[1], v_est[0])
        for dm in [0.5, 0.8, 1.0, 1.2, 1.5]:
            for da in [-np.pi/6, 0, np.pi/6]:
                v = v_mag * dm * np.array([np.cos(v_angle + da), np.sin(v_angle + da)])
                flat = np.concatenate([pair_centers.flatten(), v])
                dumbbell_starts.append(flat)
        
        # Evaluate dumbbell starts
        scored_dumbbell = []
        for flat in dumbbell_starts:
            if left() <= 5.0:
                break
            try:
                sc = _score_dumbbell(flat)
                if sc > 0:
                    scored_dumbbell.append((sc, flat.copy()))
            except Exception:
                pass
        
        scored_dumbbell.sort(key=lambda x: -x[0])
        
        # Polish best dumbbell starts with local optimization
        for sc, flat in scored_dumbbell[:5]:
            if left() <= 8.0:
                break
            try:
                opts = {'maxiter': 15000, 'fatol': 1e-13, 'xatol': 1e-10,
                        'adaptive': True, 'maxfev': 20000}
                res = scipy.optimize.minimize(_neg_score_dumbbell, flat, method='Nelder-Mead', options=opts)
                opt_flat = _normalize_dumbbell(res.x)
                sc_opt = _score_dumbbell(opt_flat)
                if sc_opt > best_score:
                    # Convert back to point representation
                    pc = opt_flat[:14].reshape(7, 2)
                    v = opt_flat[14:16]
                    pts = _dumbbell_config(pc, v)
                    pts = _normalize(pts)
                    best_score, best_pts = sc_opt, pts.copy()
            except Exception:
                pass
            
            if left() <= 10.0:
                break
            try:
                opts = {'maxiter': 5000, 'fatol': 1e-13, 'xatol': 1e-11,
                        'adaptive': True, 'maxfev': 15000}
                res = scipy.optimize.minimize(_neg_score_dumbbell, flat, method='Powell', options=opts)
                opt_flat = _normalize_dumbbell(res.x)
                sc_opt = _score_dumbbell(opt_flat)
                if sc_opt > best_score:
                    pc = opt_flat[:14].reshape(7, 2)
                    v = opt_flat[14:16]
                    pts = _dumbbell_config(pc, v)
                    pts = _normalize(pts)
                    best_score, best_pts = sc_opt, pts.copy()
            except Exception:
                pass

    # Phase 4: Final local polish on best result
    if left() > 5.0:
        try:
            opts = {'maxiter': 30000, 'fatol': 1e-14, 'xatol': 1e-11,
                    'adaptive': True, 'maxfev': 30000}
            res = scipy.optimize.minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead', options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    return _normalize(best_pts)
