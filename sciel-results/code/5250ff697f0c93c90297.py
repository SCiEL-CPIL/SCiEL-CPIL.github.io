# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_N = 20
_DEADLINE_MARGIN = 30.0  # seconds before hard limit


def _vertex_scores(pts):
    """Compute per-vertex score for a (20,2) array of points.
    Returns array of 20 scores in [0,1]."""
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    scores = np.zeros(n)
    for i in range(n):
        # distances from vertex i to others (exclude self)
        di = np.delete(d[i], i)
        # check any < 0.05
        if np.any(di < 0.05):
            scores[i] = 0.0
            continue
        di_sorted = np.sort(di)
        # gaps
        gaps = di_sorted[1:] - di_sorted[:-1]
        # first index minimizing gap
        k = int(np.argmin(gaps))
        D = (di_sorted[k] + di_sorted[k + 1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            sel = di_sorted[0:4]
        elif k == 17:
            sel = di_sorted[15:19]
        else:
            sel = di_sorted[k-1:k+3]
        # raw = mean of max(d/D, D/d)
        ratio = np.maximum(sel / D, D / sel)
        raw = np.mean(ratio)
        scores[i] = min(1.0, 1.0 / raw)
    return scores


def _total_score(pts):
    """Minimum vertex score (the metric)."""
    return float(np.min(_vertex_scores(pts)))


def _smooth_objective(params, n=_N):
    """Smooth surrogate objective to minimize.
    params: 2n angles (radius fixed at 1, or we optimize radius too? keep radius=1)
    Actually optimize angles only, radius=1.
    We use a soft-min over vertex raws to make it smooth.
    Returns a scalar to minimize (lower = better, corresponds to higher score)."""
    angles = params
    # Ensure angles are sorted and span 2*pi for convexity
    angles = np.sort(angles)
    # Normalize to span 2*pi
    span = angles[-1] - angles[0]
    if span <= 0:
        return 1e6
    angles_norm = angles[0] + (angles - angles[0]) * (2.0 * np.pi / span)
    x = np.cos(angles_norm)
    y = np.sin(angles_norm)
    pts = np.stack([x, y], axis=1)
    
    # Compute raws for each vertex (smooth version)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    
    raws = []
    for i in range(n):
        di = np.delete(d[i], i)
        if np.any(di < 0.05):
            raws.append(100.0)
            continue
        di_sorted = np.sort(di)
        gaps = di_sorted[1:] - di_sorted[:-1]
        k = int(np.argmin(gaps))
        D = (di_sorted[k] + di_sorted[k + 1]) / 2.0
        if D < 0.1:
            raws.append(100.0)
            continue
        if k == 0:
            sel = di_sorted[0:4]
        elif k == 17:
            sel = di_sorted[15:19]
        else:
            sel = di_sorted[k-1:k+3]
        ratio = np.maximum(sel / D, D / sel)
        raw = np.mean(ratio)
        raws.append(raw)
    
    raws = np.array(raws)
    # Soft max of raws (we want to minimize the max raw)
    # Use log-sum-exp for smooth max
    m = np.max(raws)
    if m > 50:
        return m
    log_sum = m + np.log(np.sum(np.exp(raws - m)))
    return log_sum


def _regular_polygon():
    angles = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    
    # Start with regular polygon
    best_pts = _regular_polygon()
    best_score = _total_score(best_pts)
    
    # Try local optimization to improve
    base_angles = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
    
    # Multi-start Nelder-Mead on angle perturbations
    n_starts = 5
    rng = np.random.default_rng(42)
    
    for start_idx in range(n_starts):
        if time.time() - start_time > 600.0 - _DEADLINE_MARGIN:
            break
        
        if start_idx == 0:
            x0 = base_angles.copy()
        else:
            # Perturb angles
            perturbation = rng.normal(0, 0.05, size=_N)
            x0 = base_angles + perturbation
        
        try:
            res = minimize(
                _smooth_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8,
                         'adaptive': True}
            )
            
            if time.time() - start_time > 600.0 - _DEADLINE_MARGIN:
                break
            
            # Convert result to points
            angles = np.sort(res.x)
            span = angles[-1] - angles[0]
            if span <= 0:
                continue
            angles_norm = angles[0] + (angles - angles[0]) * (2.0 * np.pi / span)
            x = np.cos(angles_norm)
            y = np.sin(angles_norm)
            pts = np.stack([x, y], axis=1)
            
            # Check validity
            if not np.all(np.isfinite(pts)):
                continue
            
            score = _total_score(pts)
            if score > best_score + 1e-12:
                best_score = score
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    # Final: ensure the result is valid and return
    # Re-verify best_pts
    if not np.all(np.isfinite(best_pts)):
        best_pts = _regular_polygon()
    
    return best_pts


# EVOLVE-BLOCK-END
