import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    
    def get_diameter(pts):
        # Compute pairwise distances
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        return np.max(dists)

    def objective(params):
        pts = params.reshape(n, 2)
        # Calculate pairwise distances
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        # Avoid log(0)
        dists[~np.isfinite(dists)] = 1e-10
        # Set diagonal to 1 to ignore self-distance
        np.fill_diagonal(dists, 1.0)
        
        # Log product of all ordered pairs
        # Sum of log(dists) over i != j
        log_prod = np.sum(np.log(dists))
        
        # Diameter
        D = np.max(dists)
        if D < 1e-9:
            return 1e10 # Penalty for collapse
        
        # Score is log_prod - 56 * log(D)
        # We want to maximize this, so minimize the negative
        return -(log_prod - 56 * np.log(D))

    def optimize_from_start(x0):
        try:
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 500, 'maxfev': 10000})
            return res.x
        except Exception:
            return x0

    best_score = -np.inf
    best_points = None
    
    # Strategy 1: Regular Octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts1 = np.column_stack((np.cos(angles), np.sin(angles)))
    d = get_diameter(pts1)
    pts1 = pts1 * (2.0 / d)
    res1 = optimize_from_start(pts1.flatten())
    score1 = -objective(res1)
    if score1 > best_score:
        best_score = score1
        best_points = res1

    # Strategy 2: Random Points
    for i in range(5):
        np.random.seed(42 + i) # Reproducibility
        pts_rand = np.random.randn(n, 2)
        d = get_diameter(pts_rand)
        pts_rand = pts_rand * (2.0 / d)
        res_rand = optimize_from_start(pts_rand.flatten())
        score_rand = -objective(res_rand)
        if score_rand > best_score:
            best_score = score_rand
            best_points = res_rand

    # Strategy 3: Symmetric "Cross" - 4 points on axes, 4 on diagonals
    # Parameters: r1 (axis radius), r2 (diag radius), angle offset
    # This might be overkill, let's just use a perturbed regular octagon
    # or a configuration of two squares.
    
    # Let's try a configuration of two squares, one rotated by 45 degrees
    # but with different radii.
    # Points: (r1, 0), (0, r1), (-r1, 0), (0, -r1)
    #         (r2/sqrt(2), r2/sqrt(2)), etc.
    # This is effectively a regular octagon if r1=r2.
    # Let's allow r1 != r2.
    
    def gen_two_squares(r1, r2):
        p1 = np.array([[r1, 0], [0, r1], [-r1, 0], [0, -r1]])
        p2 = np.array([[r2/np.sqrt(2), r2/np.sqrt(2)], 
                       [-r2/np.sqrt(2), r2/np.sqrt(2)], 
                       [-r2/np.sqrt(2), -r2/np.sqrt(2)], 
                       [r2/np.sqrt(2), -r2/np.sqrt(2)]])
        return np.vstack((p1, p2))
    
    for r1 in [1.0, 1.2, 0.8]:
        for r2 in [1.0, 1.2, 0.8]:
            pts_sq = gen_two_squares(r1, r2)
            d = get_diameter(pts_sq)
            pts_sq = pts_sq * (2.0 / d)
            res_sq = optimize_from_start(pts_sq.flatten())
            score_sq = -objective(res_sq)
            if score_sq > best_score:
                best_score = score_sq
                best_points = res_sq

    # Ensure distinct points and valid output
    if best_points is None:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles))).flatten()

    # Final scaling to diameter 2 is handled by the evaluator, 
    # but we should return well-scaled points to be safe.
    pts_final = best_points.reshape(n, 2)
    d = get_diameter(pts_final)
    if d > 0:
        pts_final = pts_final * (2.0 / d)
        
    return [[float(p[0]), float(p[1])] for p in pts_final]
