"""Packing 300 equal circles in a unit circle using scipy least_squares."""
import numpy as np
import time
from scipy.optimize import least_squares

N = 300
TIME_LIMIT = 115.0  # safety margin before 150s kill


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    # Target radius ~0.053, so spacing ~ 2*0.053 = 0.106
    # But let's start a bit tighter to allow optimization to expand
    # Actually, for initialization, a standard hex lattice with spacing ~0.10 is good
    spacing = 0.10
    k = int(1.1 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # Fallback: try larger grid or random
        return _random_init(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _random_init(n=N):
    """Random points inside unit disk."""
    rng = np.random.default_rng(42)
    while True:
        pts = rng.uniform(-1, 1, size=(n, 2))
        r = np.hypot(pts[:, 0], pts[:, 1])
        if np.all(r <= 1.0):
            break
        pts[r > 1.0] = pts[r > 1.0] / r[r > 1.0, None]
    return pts


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    # Efficient min pair distance
    # For N=300, O(N^2) is 45k, which is fine
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _residuals(x):
    """
    Residuals for least_squares.
    We want to maximize the minimum distance.
    A common smooth proxy is minimizing the sum of squared inverse distances,
    or minimizing a penalty on the minimum distance.
    
    Let's use: residuals = [ (r_target - d_ij)_+ / scale for all pairs, (r_target - d_boundary)_+ / scale ]
    But (.)_+ is not smooth.
    
    Alternative: Minimize sum(1/d_ij^2) + sum(1/d_bound^2).
    This strongly penalizes small distances.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    
    iu = np.triu_indices(N, 1)
    d2_pairs = dist2[iu]
    
    # Avoid zero division
    d2_pairs = np.maximum(d2_pairs, 1e-12)
    
    # Boundary distances
    rad = np.hypot(pts[:, 0], pts[:, 1])
    d_bound = 1.0 - rad
    d_bound = np.maximum(d_bound, 1e-12)
    
    # Objective: minimize sum(1/d^2)
    # This is equivalent to minimizing the sum of squared inverse distances
    r_pairs = 1.0 / d2_pairs
    r_bound = 1.0 / (d_bound * d_bound)
    
    # Scale boundary to be comparable to pairs?
    # There are N pairs vs N boundaries.
    # Let's just return them.
    return np.concatenate((r_pairs, r_bound))


def construct():
    start_time = time.time()
    
    # Initialize
    pts = _hex_init(N)
    
    # Ensure inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
        
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Optimize
    # We want to maximize the minimum distance.
    # least_squares minimizes sum(residuals^2).
    # If residuals are 1/d, minimizing sum(1/d^2) pushes all d up.
    
    x0 = pts.flatten()
    
    try:
        # Use 'trf' (trust-region reflective) for robustness
        # Bounds: x in [-1, 1], y in [-1, 1]
        # But we need x^2+y^2 <= 1.
        # least_squares doesn't handle general nonlinear constraints well directly.
        # 'trf' supports bounds. If we bound to square [-1,1], points might go outside circle.
        # We can add a penalty in residuals for boundary violations.
        
        def residuals_with_bound(x):
            pts = x.reshape(-1, 2)
            rad = np.hypot(pts[:, 0], pts[:, 1])
            
            # Pairwise
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist2, np.inf)
            iu = np.triu_indices(N, 1)
            d2_pairs = dist2[iu]
            d2_pairs = np.maximum(d2_pairs, 1e-12)
            
            # Boundary: penalty if rad > 1
            # Smooth penalty: max(0, rad - 1)^2
            # Or just 1/(1-rad) which goes to inf at boundary.
            # Let's stick to the 1/d formulation but ensure d_bound is positive.
            # If rad > 1, d_bound is negative. 1/negative is bad.
            # So we must enforce rad <= 1.
            
            # Let's use a penalty: (rad - 1)^2 * big_weight if rad > 1
            # And 1/d2_pairs for separation.
            
            bound_penalty = np.zeros(N)
            mask = rad > 1.0
            bound_penalty[mask] = (rad[mask] - 1.0) ** 2 * 100.0
            
            r_pairs = 1.0 / d2_pairs
            r_bound = bound_penalty
            
            return np.concatenate((r_pairs, r_bound))
        
        res = least_squares(
            residuals_with_bound,
            x0,
            method='trf',
            bounds=([-1.0] * (2 * N), [1.0] * (2 * N)),
            max_nfev=500,  # Limit function evaluations to stay within time
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8
        )
        
        if res.success or res.fun < np.inf:
            opt_pts = res.x.reshape(-1, 2)
            
            # Ensure points are inside the unit circle (clip if slightly out due to numeric error)
            rad = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
            out = rad > 1.0
            if np.any(out):
                opt_pts[out] = opt_pts[out] / rad[out, None]
                
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
                
    except Exception:
        pass
    
    # Second attempt with different initialization or strategy if time permits
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 30:
        # Try random init
        pts2 = _random_init(N)
        rad = np.hypot(pts2[:, 0], pts2[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts2[out] = pts2[out] / rad[out, None]
            
        x0_2 = pts2.flatten()
        try:
            res2 = least_squares(
                residuals_with_bound,
                x0_2,
                method='trf',
                bounds=([-1.0] * (2 * N), [1.0] * (2 * N)),
                max_nfev=500,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8
            )
            
            if res2.success or res2.fun < np.inf:
                opt_pts2 = res2.x.reshape(-1, 2)
                rad = np.hypot(opt_pts2[:, 0], opt_pts2[:, 1])
                out = rad > 1.0
                if np.any(out):
                    opt_pts2[out] = opt_pts2[out] / rad[out, None]
                    
                opt_score2 = _score(opt_pts2)
                if opt_score2 > best_score:
                    best_pts = opt_pts2
                    best_score = opt_score2
        except Exception:
            pass

    # Final safety
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]

    return best_pts
