# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 620.0

def _vertex_score(distances):
    d = np.sort(distances)
    if np.any(d < 0.05):
        return 0.0
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k - 1:k + 3]
    ratio = np.maximum(sel / D, D / sel)
    raw = np.mean(ratio)
    return min(1.0, 1.0 / raw)


def _min_vertex_score(polygon):
    n = len(polygon)
    scores = []
    for i in range(n):
        dists = np.linalg.norm(polygon - polygon[i], axis=1)
        dists = np.delete(dists, i)
        scores.append(_vertex_score(dists))
    return float(np.min(scores))


def _is_valid(polygon, n=20):
    if not np.all(np.isfinite(polygon)):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-12:
                return False
    pts = np.vstack([polygon, polygon[:1]])
    edges = pts[1:] - pts[:-1]
    if np.any(np.linalg.norm(edges, axis=1) < 1e-12):
        return False
    cross = np.cross(edges[:-1], edges[1:])
    if np.any(np.abs(cross) < 1e-12):
        return False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    dangles = np.diff(angles)
    dangles = np.append(dangles, angles[0] - angles[-1])
    dangles = np.arctan2(np.sin(dangles), np.cos(dangles))
    total = np.sum(dangles)
    if abs(abs(total) - 2 * np.pi) > 1e-6:
        return False
    return True


def _normalized(polygon):
    c = np.mean(polygon, axis=0)
    p = polygon - c
    m = np.mean(np.linalg.norm(p, axis=1))
    if m <= 0.01:
        return None
    return p / m


def _param_to_polygon(params):
    """Convert 40 params (20 angles, 20 radii) to polygon vertices."""
    angles = np.sort(params[:20])
    radii = params[20:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _polygon_to_params(polygon):
    """Convert polygon to (angles, radii) params."""
    c = np.mean(polygon, axis=0)
    p = polygon - c
    angles = np.arctan2(p[:, 1], p[:, 0])
    angles = np.sort(angles)
    radii = np.linalg.norm(p, axis=1)
    # Reorder radii to match sorted angles
    order = np.argsort(np.arctan2(polygon[:, 1] - c[1], polygon[:, 0] - c[0]))
    radii = radii[order]
    return np.concatenate([angles, radii])


def _neg_min_score(params):
    """Objective: negative of min vertex score (to maximize)."""
    try:
        polygon = _param_to_polygon(params)
        if not _is_valid(polygon):
            return 1.0  # large penalty
        nm = _normalized(polygon)
        if nm is None:
            return 1.0
        score = _min_vertex_score(nm)
        return -score
    except Exception:
        return 1.0


def _optimize_from_start(start_params, n_iter=200, timeout=30):
    """Run L-BFGS-B optimization from start_params."""
    t0 = time.time()
    try:
        result = minimize(
            _neg_min_score,
            start_params,
            method='L-BFGS-B',
            options={'maxiter': n_iter, 'maxfun': 5000}
        )
        if time.time() - t0 > timeout:
            return start_params, _neg_min_score(start_params)
        if result.fun < _neg_min_score(start_params):
            return result.x, result.fun
        return start_params, _neg_min_score(start_params)
    except Exception:
        return start_params, _neg_min_score(start_params)


def equidistant_points_polygon20() -> np.ndarray:
    # Start with regular 20-gon
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    reg = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
    best = reg.copy()
    best_score = _min_vertex_score(_normalized(reg))
    
    # Get initial params
    init_params = _polygon_to_params(reg)
    
    # Try multiple optimization runs with different perturbations
    rng = np.random.default_rng(42)
    
    # First, try direct optimization from regular polygon
    if time.time() < _DEADLINE - 60:
        opt_params, opt_val = _optimize_from_start(init_params, n_iter=300, timeout=50)
        if opt_val < -best_score:
            best_score = -opt_val
            best = _normalized(_param_to_polygon(opt_params))
            if best is not None:
                best = best
    
    # Try perturbations and optimize from each
    for trial in range(50):
        if time.time() > _DEADLINE - 30:
            break
        
        # Perturb the regular polygon
        perturb_scale = 0.05 * (1 + trial * 0.1)
        perturb = rng.normal(0, perturb_scale, size=(n, 2))
        trial_poly = reg + perturb
        
        if not _is_valid(trial_poly):
            continue
        
        nm = _normalized(trial_poly)
        if nm is None:
            continue
        
        s = _min_vertex_score(nm)
        if s > best_score:
            best_score = s
            best = nm
        
        # Optimize from this perturbed start
        if time.time() < _DEADLINE - 40:
            trial_params = _polygon_to_params(trial_poly)
            opt_params, opt_val = _optimize_from_start(trial_params, n_iter=100, timeout=15)
            if opt_val < -best_score:
                opt_poly = _param_to_polygon(opt_params)
                nm_opt = _normalized(opt_poly)
                if nm_opt is not None and _is_valid(nm_opt):
                    s_opt = _min_vertex_score(nm_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best = nm_opt
    
    # Final local refinement
    if time.time() < _DEADLINE - 10 and best is not None:
        final_params = _polygon_to_params(best)
        opt_params, opt_val = _optimize_from_start(final_params, n_iter=200, timeout=20)
        if opt_val < -best_score:
            opt_poly = _param_to_polygon(opt_params)
            nm_opt = _normalized(opt_poly)
            if nm_opt is not None and _is_valid(nm_opt):
                s_opt = _min_vertex_score(nm_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best = nm_opt
    
    return best

# EVOLVE-BLOCK-END
