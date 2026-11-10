import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-14:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _fourier_to_points(coeffs, n=20):
    """Convert Fourier coefficients to polygon points.
    coeffs: array of shape (2, K) where K is number of harmonics.
    coeffs[0, k] = amplitude of cos(k*theta), coeffs[1, k] = amplitude of sin(k*theta)
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # Start with unit circle, add perturbations
    r = np.ones(n)
    for k in range(1, len(coeffs[0])):
        r += coeffs[0, k] * np.cos(k * angles) + coeffs[1, k] * np.sin(k * angles)
    # Ensure positive radii for star-shaped polygon
    r = np.maximum(r, 0.5)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _soft_min(scores, temperature=0.01):
    """Smoothed minimum using log-sum-exp."""
    min_val = np.min(scores)
    shifted = scores - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum

def _smooth_objective(coords_flat, temperature=0.01):
    """Smoothed objective: negative soft-min of vertex scores."""
    n = 20
    pts = coords_flat.reshape(n, 2)
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 1000.0
    pts_norm = pts_norm / mean_dist
    
    # Check convexity
    if not _is_valid(pts_norm):
        return 1000.0
    
    # Compute vertex scores
    scores = np.array([_vertex_score(pts_norm[i], pts_norm) for i in range(n)])
    
    # Use soft-min to make objective smooth
    soft_min_val = _soft_min(scores, temperature)
    
    return -soft_min_val

def _objective(coeffs):
    """Objective function: negative min vertex score (we minimize)."""
    pts = _fourier_to_points(coeffs)
    pts_norm, ok = _normalize(pts)
    if not ok:
        return 10.0
    if not _is_valid(pts_norm):
        return 10.0
    score = _min_vertex_score(pts_norm)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Regular 20-gon as baseline
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    pts_norm_regular, ok = _normalize(pts_regular)
    if ok and _is_valid(pts_norm_regular):
        best_score = _min_vertex_score(pts_norm_regular)
        best_pts = pts_norm_regular
    else:
        best_score = 0.0
        best_pts = pts_regular
    
    # Try optimization with Fourier coefficients
    # Use 3 harmonics (6 coefficients)
    K = 3
    bounds = [(-0.3, 0.3)] * (2 * K)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_pts
    
    try:
        # Use differential evolution
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return best_pts
        
        if result.fun < -best_score:
            coeffs_opt = result.x
            pts_opt = _fourier_to_points(coeffs_opt)
            pts_norm_opt, ok = _normalize(pts_opt)
            if ok and _is_valid(pts_norm_opt):
                score_opt = _min_vertex_score(pts_norm_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_norm_opt
                    
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B with smoothed objective in coordinate space
    try:
        coords_flat = best_pts.flatten()
        
        # Multiple temperature values for annealing
        temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
        
        for temp in temperatures:
            if time.time() > deadline - 60.0:
                break
            
            def obj(coords_flat):
                return _smooth_objective(coords_flat, temperature=temp)
            
            result = minimize(obj, coords_flat, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < -best_score + 0.001:
                candidate = result.x.reshape(20, 2)
                candidate, valid = _normalize(candidate)
                if valid and _is_valid(candidate):
                    score = _min_vertex_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate
                        coords_flat = candidate.flatten()
                        
    except Exception:
        pass
    
    # Phase 3: Simulated annealing in coordinate space
    try:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.1
        T_min = 0.001
        cooling_rate = 0.95
        max_iter = 5000
        
        for i in range(max_iter):
            if time.time() > deadline - 30.0:
                break
            
            # Random perturbation
            perturbation = np.random.normal(0, T * 0.5, (20, 2))
            candidate = current_pts + perturbation
            
            # Normalize and check validity
            candidate, valid = _normalize(candidate)
            if not valid or not _is_valid(candidate):
                continue
            
            candidate_score = _min_vertex_score(candidate)
            
            # Accept if better, or with probability exp(-delta/T)
            if candidate_score > current_score:
                current_pts = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = candidate.copy()
            else:
                delta = current_score - candidate_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_pts = candidate
                        current_score = candidate_score
            
            T *= cooling_rate
            if T < T_min:
                break
                
    except Exception:
        pass
    
    # Phase 4: Final local refinement with tight tolerance
    try:
        if time.time() < deadline - 10.0:
            coords_flat = best_pts.flatten()
            
            def obj_tight(coords_flat):
                return _smooth_objective(coords_flat, temperature=0.001)
            
            result = minimize(obj_tight, coords_flat, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < -best_score + 0.0001:
                candidate = result.x.reshape(20, 2)
                candidate, valid = _normalize(candidate)
                if valid and _is_valid(candidate):
                    score = _min_vertex_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate
                        
    except Exception:
        pass
    
    return best_pts
