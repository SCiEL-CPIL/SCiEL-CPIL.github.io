import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def _diameter(pts):
    """Compute the diameter (max pairwise distance) of a set of points."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    return np.max(dists)

def _log_product(pts):
    """Compute log of product of all pairwise distances (i<j)."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-14:
        return -np.inf
    return np.sum(np.log(d))

def _score_from_log_product_and_diameter(log_prod, D):
    """
    Score after scaling to diameter 2.
    Each distance scales by 2/D, so product scales by (2/D)^28.
    log_score = log_prod + 28*log(2) - 28*log(D)
    The constant 28*log(2) doesn't affect relative comparison.
    """
    if D < 1e-14:
        return -np.inf
    return log_prod - 28.0 * np.log(D)

def _score(pts):
    """Compute the score for a set of 8 points."""
    n = len(pts)
    if n != 8:
        return -np.inf
    # Check distinctness
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(8, k=1)
    d = dists[iu]
    if np.min(d) < 1e-10:
        return -np.inf
    
    D = _diameter(pts)
    log_prod = _log_product(pts)
    if np.isinf(log_prod) or np.isnan(log_prod):
        return -np.inf
    return _score_from_log_product_and_diameter(log_prod, D)

def _objective(x):
    """
    Objective for minimization. x is 12 values: coordinates of points 2..7.
    Points 0 and 1 are fixed at (-1,0) and (1,0).
    """
    pts = np.zeros((8, 2))
    pts[0] = [-1.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:, 0] = x[0::2]
    pts[2:, 1] = x[1::2]
    
    # Penalty if any two points coincide or are too close
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(8, k=1)
    d = dists[iu]
    if np.min(d) < 1e-10:
        return 1e10
    
    s = _score(pts)
    if np.isinf(s) or np.isnan(s):
        return 1e10
    return -s

def _refine(pts):
    """
    Refine a point set by fixing the diameter pair and using L-BFGS-B.
    Returns the refined point set.
    """
    # Find the diameter pair
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    if i > j:
        i, j = j, i
    
    # Translate and rotate so that pts[i] -> (-1,0) and pts[j] -> (1,0)
    mid = (pts[i] + pts[j]) / 2.0
    vec = pts[j] - pts[i]
    d_ij = np.linalg.norm(vec)
    if d_ij < 1e-12:
        return pts
    
    angle = np.arctan2(vec[1], vec[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    scaled = (pts - mid) * (2.0 / d_ij)
    rotated = scaled @ rot_mat.T
    
    # Now rotated[i] should be (-1,0) and rotated[j] should be (1,0)
    other_indices = [k for k in range(8) if k != i and k != j]
    x0 = rotated[other_indices].flatten()
    
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 1000, 'maxfun': 10000, 'gtol': 1e-10})
        
        if res.fun < 1e9:
            new_pts = np.zeros((8, 2))
            new_pts[i] = [-1.0, 0.0]
            new_pts[j] = [1.0, 0.0]
            new_pts[other_indices, 0] = res.x[0::2]
            new_pts[other_indices, 1] = res.x[1::2]
            return new_pts
    except Exception:
        pass
    
    return pts

def _make_cage_init(rng):
    """
    Create a cage-like initial configuration.
    The optimal configuration for maximizing product of distances tends to have
    points arranged in a symmetric pattern. We start with a perturbed regular octagon
    or a structured configuration.
    """
    # Regular octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Add small random perturbation
    perturbed = oct_pts + rng.normal(0, 0.1, size=(8, 2))
    return perturbed

def _make_double_square_init(rng, r2=0.7):
    """Create a double square initial configuration."""
    sq1 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    sq2 = np.array([[r2 * np.cos(np.pi/4), r2 * np.sin(np.pi/4)],
                    [r2 * np.cos(3*np.pi/4), r2 * np.sin(3*np.pi/4)],
                    [r2 * np.cos(5*np.pi/4), r2 * np.sin(5*np.pi/4)],
                    [r2 * np.cos(7*np.pi/4), r2 * np.sin(7*np.pi/4)]], dtype=float)
    pts = np.vstack([sq1, sq2])
    pts = pts + rng.normal(0, 0.05, size=(8, 2))
    return pts

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s for overhead/return
    
    rng = np.random.default_rng(42)
    
    # ---- Initial guesses ----
    initial_sets = []
    
    # 1. Regular octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_sets.append(oct_pts.copy())
    
    # 2. Double square with r2=0.7
    initial_sets.append(_make_double_square_init(rng, 0.7))
    
    # 3. Double square with r2=0.5
    initial_sets.append(_make_double_square_init(rng, 0.5))
    
    # 4. Double square with r2=0.8
    initial_sets.append(_make_double_square_init(rng, 0.8))
    
    # 5. Random perturbed octagon
    initial_sets.append(oct_pts + rng.normal(0, 0.2, size=(8, 2)))
    
    # Evaluate initial sets
    best_score = -np.inf
    best_pts = oct_pts.copy()
    
    for pts in initial_sets:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # ---- Optimization loop with time budget ----
    max_restarts = 300
    restart = 0
    
    while time.time() < deadline and restart < max_restarts:
        # Generate initial guess
        if restart < len(initial_sets):
            base = initial_sets[restart].copy()
        elif restart % 3 == 0:
            # Perturb the best found so far
            base = best_pts + rng.normal(0, 0.1, size=(8, 2))
        elif restart % 3 == 1:
            # Random octagon with perturbation
            base = oct_pts + rng.normal(0, 0.15, size=(8, 2))
        else:
            # Double square with random ratio
            r2 = 0.4 + 0.5 * rng.random()
            base = _make_double_square_init(rng, r2)
        
        # Ensure distinct points
        for _ in range(50):
            diffs = base[:, np.newaxis, :] - base[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            np.fill_diagonal(dists, 1.0)
            iu = np.triu_indices(8, k=1)
            if np.min(dists[iu]) > 1e-8:
                break
            base += rng.normal(0, 0.01, size=(8, 2))
        
        # Refine using L-BFGS-B
        refined = _refine(base)
        
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_pts = refined.copy()
        
        restart += 1
    
    # ---- Final: scale to diameter 2 and return ----
    pts = best_pts.copy()
    D = _diameter(pts)
    if D > 1e-12:
        pts = pts * (2.0 / D)
    
    # Verify distinct
    for i in range(8):
        for j in range(i + 1, 8):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                pts[j] += 1e-8
    
    return [[float(p[0]), float(p[1])] for p in pts]
