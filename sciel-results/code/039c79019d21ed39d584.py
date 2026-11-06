import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_score(pts):
        """Compute the score for a set of points."""
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def score_from_points(pts):
        return compute_score(pts)

    # Family 1: Two Concentric Squares
    # Params: r1, r2, theta
    # Points:
    # P1: r1 * (cos(0), sin(0)), r1 * (cos(pi/2), sin(pi/2)), ...
    # P2: r2 * (cos(theta), sin(theta)), ...
    def get_two_squares(r1, r2, theta):
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def neg_score_two_squares(params):
        r1, r2, theta = params
        if r1 < 0 or r2 < 0:
            return 1e10
        pts = get_two_squares(r1, r2, theta)
        s = compute_score(pts)
        return -s

    # Family 2: Two Concentric Rectangles
    # Params: w1, h1, w2, h2, theta
    # Points:
    # P1: (w1, h1), (-w1, h1), (-w1, -h1), (w1, -h1)
    # P2: (w2 cos(t) - h2 sin(t), w2 sin(t) + h2 cos(t)), ...
    def get_two_rects(w1, h1, w2, h2, theta):
        # Rect 1
        pts1 = np.array([
            [w1, h1],
            [-w1, h1],
            [-w1, -h1],
            [w1, -h1]
        ])
        # Rect 2
        c, s = np.cos(theta), np.sin(theta)
        p0 = np.array([w2, h2])
        p1 = np.array([-w2, h2])
        p2 = np.array([-w2, -h2])
        p3 = np.array([w2, -h2])
        
        def rot(p):
            return np.array([c*p[0] - s*p[1], s*p[0] + c*p[1]])
            
        pts2 = np.array([rot(p0), rot(p1), rot(p2), rot(p3)])
        return np.vstack([pts1, pts2])

    def neg_score_two_rects(params):
        w1, h1, w2, h2, theta = params
        if w1 < 0 or h1 < 0 or w2 < 0 or h2 < 0:
            return 1e10
        pts = get_two_rects(w1, h1, w2, h2, theta)
        s = compute_score(pts)
        return -s

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        if not np.all(np.isfinite(pts)):
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    try_candidate(reg_pts)

    # 2. Optimize Two Concentric Squares
    np.random.seed(42)
    
    # Grid search / Multistart for 3 params
    # r1, r2 in [0.1, 1.5], theta in [0, pi/2]
    # We'll use Powell for robustness
    
    best_params_sq = None
    best_val_sq = 1e10
    
    # Initial guesses
    inits_sq = [
        [1.0, 1.0, np.pi/4], # Regular Octagon
        [0.5, 1.0, np.pi/4],
        [1.0, 0.5, np.pi/4],
        [0.8, 0.8, 0.0],
        [0.8, 0.8, np.pi/2],
        [0.6, 1.2, np.pi/8],
        [1.2, 0.6, np.pi/8],
    ]
    
    for _ in range(20):
        r1 = 0.3 + 1.2 * np.random.rand()
        r2 = 0.3 + 1.2 * np.random.rand()
        theta = np.random.rand() * np.pi/2
        inits_sq.append([r1, r2, theta])

    for guess in inits_sq:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(
                neg_score_two_squares,
                guess,
                method='Powell',
                bounds=[(0.01, 2.0), (0.01, 2.0), (0.0, np.pi/2)],
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.fun < best_val_sq:
                best_val_sq = res.fun
                best_params_sq = res.x
        except Exception:
            pass

    if best_params_sq is not None:
        pts = get_two_squares(*best_params_sq)
        try_candidate(pts)

    # 3. Optimize Two Concentric Rectangles
    best_params_rect = None
    best_val_rect = 1e10
    
    inits_rect = [
        [1.0, 1.0, 1.0, 1.0, np.pi/4], # Two squares, rotated
        [1.0, 0.5, 1.0, 0.5, np.pi/4],
        [1.0, 1.0, 0.5, 0.5, np.pi/4],
        [0.8, 0.8, 0.8, 0.8, 0.0],
    ]
    
    for _ in range(20):
        w1 = 0.3 + 1.2 * np.random.rand()
        h1 = 0.3 + 1.2 * np.random.rand()
        w2 = 0.3 + 1.2 * np.random.rand()
        h2 = 0.3 + 1.2 * np.random.rand()
        theta = np.random.rand() * np.pi/2
        inits_rect.append([w1, h1, w2, h2, theta])

    for guess in inits_rect:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(
                neg_score_two_rects,
                guess,
                method='Powell',
                bounds=[(0.01, 2.0), (0.01, 2.0), (0.01, 2.0), (0.01, 2.0), (0.0, np.pi/2)],
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.fun < best_val_rect:
                best_val_rect = res.fun
                best_params_rect = res.x
        except Exception:
            pass

    if best_params_rect is not None:
        pts = get_two_rects(*best_params_rect)
        try_candidate(pts)

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Scale to ensure coords are small
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
