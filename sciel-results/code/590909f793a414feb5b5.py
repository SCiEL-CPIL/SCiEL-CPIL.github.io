# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    if signs.count(signs[0]) != n:
        return False
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:
            break
    return min_vs

def _score_polygon_fast(p):
    """Vectorized scoring for speed."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    n = len(pn)
    # Compute all pairwise distances
    diff = pn[:, np.newaxis, :] - pn[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)  # Set self-distance to large value
    min_vs = 1.0
    for i in range(n):
        d_sorted = np.sort(dists[i, :])
        # Remove self (should be at index 0 if we filled diagonal with 1.0, but actually it's at position i)
        # Actually, let's just use the 19 non-self distances
        d = np.delete(dists[i, :], i)
        d_sorted = np.sort(d)
        if np.any(d_sorted < 0.05):
            return 0.0
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = int(np.argmin(gaps))
        D = (d_sorted[k] + d_sorted[k + 1]) / 2
        if D < 0.1:
            return 0.0
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k - 1:k + 3]
        raw = np.mean([max(di / D, D / di) for di in selected])
        if raw < 1e-12:
            return 0.0
        score = min(1.0, 1.0 / raw)
        if score < min_vs:
            min_vs = score
        if min_vs < 0.01:
            return 0.0
    return min_vs

def _smooth_vertex_score(p, v_idx, beta=50.0):
    """Fully smooth surrogate for vertex score."""
    n = len(p)
    v = p[v_idx]
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    
    # Soft penalty for d < 0.05
    min_d = d_sorted[0]
    penalty_min = 1.0 / (1.0 + np.exp(beta * (min_d - 0.05)))
    
    gaps = d_sorted[1:] - d_sorted[:-1]
    neg_gaps = -gaps
    max_neg = np.max(neg_gaps)
    weights = np.exp(beta * (neg_gaps - max_neg))
    weights /= np.sum(weights)
    
    midpoints = (d_sorted[:-1] + d_sorted[1:]) / 2
    D_soft = np.sum(weights * midpoints)
    
    penalty_D = 1.0 / (1.0 + np.exp(beta * (D_soft - 0.1)))
    
    # Smoothly blend the three selection cases
    k_soft = np.sum(weights * np.arange(len(gaps)))
    
    # For each possible k, compute the raw value
    raws = np.zeros(len(gaps))
    for k_idx in range(len(gaps)):
        if k_idx == 0:
            sel = d_sorted[0:4]
        elif k_idx == 17:
            sel = d_sorted[15:19]
        else:
            sel = d_sorted[k_idx - 1:k_idx + 3]
        ratios = np.maximum(sel / D_soft, D_soft / sel)
        raws[k_idx] = np.mean(ratios)
    
    raw_soft = np.sum(weights * raws)
    
    if raw_soft < 1e-12:
        return 0.0
    
    score = min(1.0, 1.0 / raw_soft)
    return score * penalty_min * penalty_D

def _smooth_score_polygon(p, beta=50.0):
    """Smooth surrogate for polygon score using soft-min over vertices."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    scores = np.array([_smooth_vertex_score(pn, i, beta) for i in range(len(pn))])
    # Soft min
    neg_scores = -beta * scores
    max_neg = np.max(neg_scores)
    soft_min = -np.log(np.sum(np.exp(neg_scores - max_neg))) / beta - max_neg / beta
    return soft_min

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_score = 0.0
    best_pts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def try_candidate(pts):
        nonlocal best_score, best_pts
        s = _score_polygon_fast(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    try_candidate(best_pts)
    
    # Phase 1: Broad exploration with sinusoidal perturbations
    modes = [1, 2, 3, 4, 5, 7, 10]
    phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    amps_a = [0.01, 0.02, 0.05, 0.1, 0.15]
    amps_r = [0.01, 0.02, 0.05, 0.1]
    
    def make_perturbation(mode, phase, amp_a, amp_r):
        a = base_angles
        angles_p = a + amp_a * np.sin(mode * a + phase)
        radii_p = 1.0 + amp_r * np.sin(mode * a + phase * 1.3)
        pts = np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)])
        return pts
    
    for mode in modes:
        if time.time() > deadline:
            break
        for phase in phases:
            if time.time() > deadline:
                break
            for amp_a in amps_a:
                if time.time() > deadline:
                    break
                for amp_r in amps_r:
                    if time.time() > deadline:
                        break
                    pts = make_perturbation(mode, phase, amp_a, amp_r)
                    try_candidate(pts)
    
    # Phase 2: L-BFGS-B optimization on smooth surrogate
    if best_score > 0 and time.time() < deadline - 30:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        def objective(params):
            radii = params[:n]
            angles = params[n:]
            radii = np.clip(radii, 0.5, 2.0)
            pts = reconstruct(radii, angles)
            s_smooth = _smooth_score_polygon(pts, beta=30.0)
            s_exact = _score_polygon_fast(pts)
            if s_exact == 0.0:
                return -s_smooth - 1.0
            return -s_smooth
        
        x0 = np.concatenate([cur_radii, cur_angles])
        bounds = [(0.5, 2.0)] * n + [(-np.pi, np.pi)] * n
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < 0:
                opt_radii = np.clip(result.x[:n], 0.5, 2.0)
                opt_angles = result.x[n:]
                opt_pts = reconstruct(opt_radii, opt_angles)
                try_candidate(opt_pts)
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead on exact score
    if best_score > 0 and time.time() < deadline - 20:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        def objective_nm(params):
            radii = params[:n]
            angles = params[n:]
            radii = np.clip(radii, 0.5, 2.0)
            pts = reconstruct(radii, angles)
            s = _score_polygon_fast(pts)
            if s == 0.0:
                return 1.0
            return -s
        
        x0 = np.concatenate([cur_radii, cur_angles])
        
        try:
            result = minimize(objective_nm, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            if result.fun < 0:
                opt_radii = np.clip(result.x[:n], 0.5, 2.0)
                opt_angles = result.x[n:]
                opt_pts = reconstruct(opt_radii, opt_angles)
                try_candidate(opt_pts)
        except Exception:
            pass
    
    # Phase 4: Coordinate descent refinement
    if best_score > 0 and time.time() < deadline - 10:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        step = 0.01
        no_improve_count = 0
        
        while time.time() < deadline - 5 and step > 0.0005:
            improved = False
            
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for delta in [step, -step]:
                    new_angles = cur_angles.copy()
                    new_angles[i] += delta
                    new_pts = reconstruct(cur_radii, new_angles)
                    s = _score_polygon_fast(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        current = new_pts
                        cur_angles = new_angles
                        improved = True
            
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for delta in [step, -step]:
                    new_radii = cur_radii.copy()
                    new_radii[i] = max(0.5, min(2.0, new_radii[i] + delta))
                    new_pts = reconstruct(new_radii, cur_angles)
                    s = _score_polygon_fast(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        current = new_pts
                        cur_radii = new_radii
                        improved = True
            
            if not improved:
                step *= 0.7
                no_improve_count += 1
                if no_improve_count > 5:
                    break
            else:
                no_improve_count = 0
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 2:
        current = best_pts.copy()
        cur_radii = np.linalg.norm(current, axis=1)
        cur_angles = np.arctan2(current[:, 1], current[:, 0])
        
        def reconstruct(radii, angles):
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        step = 0.002
        for _ in range(30):
            if time.time() > deadline - 1:
                break
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    new_angles = cur_angles.copy()
                    new_angles[i] += delta
                    new_pts = reconstruct(cur_radii, new_angles)
                    s = _score_polygon_fast(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        cur_angles = new_angles
                        improved = True
            for i in range(n):
                for delta in [step, -step]:
                    new_radii = cur_radii.copy()
                    new_radii[i] = max(0.5, min(2.0, new_radii[i] + delta))
                    new_pts = reconstruct(new_radii, cur_angles)
                    s = _score_polygon_fast(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        cur_radii = new_radii
                        improved = True
            if not improved:
                step *= 0.7
                if step < 0.0005:
                    break
    
    return best_pts

# EVOLVE-BLOCK-END
