import numpy as np
from scipy.optimize import minimize
import time
import math

def _log_score(pts):
    """
    Fast vectorized log-score: sum_{i<j} log(d_ij) - 28*log(D/2).
    Higher is better. This is scale-invariant.
    """
    n = len(pts)
    if n != 8:
        return -np.inf
    # Pairwise distance matrix
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Extract upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-14:
        return -np.inf
    log_prod = np.sum(np.log(d))
    D = np.max(d)
    if D < 1e-14:
        return -np.inf
    # Score = product of distances / D^28, so log = log_prod - 28*log(D)
    # But we normalize to diameter 2, so effective score after scaling to diam 2:
    # each distance scales by 2/D, so log_prod_scaled = log_prod + 28*log(2/D)
    # = log_prod + 28*log(2) - 28*log(D)
    # The constant 28*log(2) doesn't affect optimization.
    # So we maximize: log_prod - 28*log(D)
    return log_prod - 28.0 * np.log(D)

def _objective_fixed(x):
    """
    Objective for minimization with p0=(-1,0), p1=(1,0) fixed.
    x is 12 values: coordinates of points 2..7.
    """
    pts = np.zeros((8, 2))
    pts[0] = [-1.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:, 0] = x[0::2]
    pts[2:, 1] = x[1::2]
    
    # Penalty if any two non-fixed points coincide or are too close to fixed points
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(8, k=1)
    d = dists[iu]
    if np.min(d) < 1e-10:
        return 1e10
    
    s = _log_score(pts)
    if np.isinf(s) or np.isnan(s):
        return 1e10
    return -s

def _objective_free(x):
    """
    Objective for minimization with all 16 coordinates free.
    Scale-invariant, so we add a small regularization to prevent drift.
    """
    pts = x.reshape(8, 2)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(8, k=1)
    d = dists[iu]
    if np.min(d) < 1e-10:
        return 1e10
    
    s = _log_score(pts)
    if np.isinf(s) or np.isnan(s):
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s for overhead/return
    
    n = 8
    
    # ---- Initial guesses ----
    
    # 1. Regular octagon (diameter 2, centered at origin)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Double square: two squares, one rotated 45° relative to the other
    #    Square 1: (1,0), (0,1), (-1,0), (0,-1)  -- these are at distance sqrt(2) from each other
    #    Actually let's make two squares at different radii or offsets
    #    Square 1: (1,0), (0,1), (-1,0), (0,-1)  (diamond, radius 1)
    #    Square 2: (0.7, 0.7), (-0.7, 0.7), (-0.7, -0.7), (0.7, -0.7)  (square, radius ~1)
    #    This is actually just the regular octagon. Let me try different radii.
    #    Square 1 at radius 1: (1,0),(0,1),(-1,0),(0,-1)
    #    Square 2 at radius 0.7, rotated 45°: (0.495, 0.495), (-0.495, 0.495), (-0.495, -0.495), (0.495, -0.495)
    sq1 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    sq2 = np.array([[0.7 * np.cos(np.pi/4), 0.7 * np.sin(np.pi/4)],
                    [0.7 * np.cos(3*np.pi/4), 0.7 * np.sin(3*np.pi/4)],
                    [0.7 * np.cos(5*np.pi/4), 0.7 * np.sin(5*np.pi/4)],
                    [0.7 * np.cos(7*np.pi/4), 0.7 * np.sin(7*np.pi/4)]], dtype=float)
    double_sq = np.vstack([sq1, sq2])
    
    # 3. Another double square with different ratio
    sq1b = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    sq2b = np.array([[0.5 * np.cos(np.pi/4), 0.5 * np.sin(np.pi/4)],
                     [0.5 * np.cos(3*np.pi/4), 0.5 * np.sin(3*np.pi/4)],
                     [0.5 * np.cos(5*np.pi/4), 0.5 * np.sin(5*np.pi/4)],
                     [0.5 * np.cos(7*np.pi/4), 0.5 * np.sin(7*np.pi/4)]], dtype=float)
    double_sq2 = np.vstack([sq1b, sq2b])
    
    # 4. Rectangle-based: two rectangles
    #    Points at (±1, ±a) and (±b, ±c)
    a, b, c = 0.6, 0.5, 0.8
    rect_pts = np.array([[1, a], [1, -a], [-1, a], [-1, -a],
                         [b, c], [b, -c], [-b, c], [-b, -c]], dtype=float)
    
    # List of initial point sets
    initial_sets = [oct_pts, double_sq, double_sq2, rect_pts]
    
    best_score = -np.inf
    best_pts = oct_pts.copy()
    
    # Evaluate initial sets
    for pts in initial_sets:
        s = _log_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # ---- Optimization loop with time budget ----
    max_restarts = 200
    restart = 0
    
    while time.time() < deadline and restart < max_restarts:
        # Choose base: alternate between fixed and free
        use_fixed = (restart % 2 == 0)
        
        # Generate initial guess
        if restart < len(initial_sets):
            base = initial_sets[restart].copy()
        else:
            # Perturb the best found so far, or a random octagon
            if restart % 3 == 0:
                base = best_pts + rng.normal(0, 0.1, size=(8, 2))
            else:
                base = oct_pts + rng.normal(0, 0.15, size=(8, 2))
        
        # Ensure distinct points
        for _ in range(100):
            diffs = base[:, np.newaxis, :] - base[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            np.fill_diagonal(dists, 1.0)
            iu = np.triu_indices(8, k=1)
            if np.min(dists[iu]) > 1e-8:
                break
            base += rng.normal(0, 0.01, size=(8, 2))
        
        try:
            if use_fixed:
                # Fix the diameter pair. Find the pair with max distance in base.
                diffs = base[:, np.newaxis, :] - base[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs ** 2, axis=2))
                i, j = np.unravel_index(np.argmax(dists), dists.shape)
                if i > j:
                    i, j = j, i
                
                # We'll fix point i to (-1,0) and point j to (1,0).
                # But the diameter pair might not be exactly at distance 2 after we move others.
                # Strategy: translate and rotate so that base[i] -> (-1,0) and base[j] -> (1,0),
                # then optimize the remaining 6 points.
                # The diameter of the final config will be >= 2, and we normalize at the end.
                
                # Translate and rotate base so that base[i] -> (-1,0), base[j] -> (1,0)
                mid = (base[i] + base[j]) / 2.0
                vec = base[j] - base[i]
                d_ij = np.linalg.norm(vec)
                if d_ij < 1e-12:
                    continue
                # Rotation: map vec to (d_ij, 0), then scale so d_ij -> 2
                angle = np.arctan2(vec[1], vec[0])
                cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                scaled = (base - mid) * (2.0 / d_ij)
                rotated = scaled @ rot_mat.T
                
                # Now rotated[i] should be (-1,0) and rotated[j] should be (1,0)
                # Extract the other 6 points
                other_indices = [k for k in range(8) if k != i and k != j]
                x0 = rotated[other_indices].flatten()
                
                res = minimize(_objective_fixed, x0, method='L-BFGS-B',
                               options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-8})
                
                if res.fun < 1e9:
                    # Reconstruct full point set
                    pts = np.zeros((8, 2))
                    pts[i] = [-1.0, 0.0]
                    pts[j] = [1.0, 0.0]
                    pts[other_indices, 0] = res.x[0::2]
                    pts[other_indices, 1] = res.x[1::2]
                    
                    s = _log_score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
            else:
                # Free optimization with all 16 coordinates
                x0 = base.flatten()
                res = minimize(_objective_free, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < 1e9:
                    pts = res.x.reshape(8, 2)
                    s = _log_score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
                
        except Exception:
            pass
        
        restart += 1
    
    # ---- Final: scale to diameter 2 and return ----
    pts = best_pts.copy()
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    D = np.max(dists)
    if D > 1e-12:
        pts = pts * (2.0 / D)
    
    # Verify distinct
    for i in range(8):
        for j in range(i + 1, 8):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                pts[j] += 1e-8
    
    return [[float(p[0]), float(p[1])] for p in pts]
