"""Program for eight planar points maximizing the product of distances at fixed diameter."""
import numpy as np
from scipy.optimize import minimize
from itertools import product

# EVOLVE-BLOCK-START
def _score_from_points(points):
    """Compute Delta / n^n for a set of n points, scaled to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    # Diameter is the max pairwise distance
    diameter = np.max(dists)
    if diameter < 1e-15:
        return 0.0
    
    # Scale so diameter is 2
    scale = 2.0 / diameter
    dists_scaled = dists * scale
    
    # Product over ordered pairs (i != j)
    # For n points, there are n*(n-1) ordered pairs
    # Delta = product of all ordered pair distances
    # log(Delta) = sum over i != j of log(d_ij)
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(dists_scaled[mask]))
    
    # Score = Delta / n^n = exp(log_delta) / n^n = exp(log_delta - n*log(n))
    score = np.exp(log_delta - n * np.log(n))
    
    return float(score)

def _generate_two_squares(r1, r2, theta):
    """Generate 8 points: two concentric squares with radii r1, r2 and relative rotation theta."""
    # First square: radius r1, rotation 0
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Second square: radius r2, rotation theta
    angles2 = angles1 + theta
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    return np.vstack([pts1, pts2])

def _negative_score(params):
    """Negative score for minimization."""
    r1, r2, theta = params
    if r1 < 0 or r2 < 0:
        return 1e10
    points = _generate_two_squares(r1, r2, theta)
    return -_score_from_points(points)

def construct():
    """Construct 8 points maximizing the product of distances at fixed diameter.
    
    Uses two concentric squares with independent radii and relative rotation.
    Performs multi-start global grid search followed by local refinement.
    """
    best_score = -np.inf
    best_params = None
    
    # By symmetry of the square, we can restrict theta to [0, pi/4]
    # and assume r1 >= r2 (or handle both cases)
    
    # Multi-start optimization: try many different initial points
    # Regular octagon is at r1 = r2, theta = pi/8
    # Try configurations near and far from this
    
    # Generate candidate initial points
    candidates = []
    
    # Regular octagon (baseline)
    candidates.append((1.0, 1.0, np.pi/8))
    
    # Grid search in a refined region
    # r1/r2 ratio from 0.5 to 2.0, theta from 0 to pi/4
    r_ratio_vals = np.linspace(0.5, 2.0, 15)
    theta_vals = np.linspace(0, np.pi/4, 15)
    r_base_vals = np.linspace(0.5, 1.5, 10)
    
    for r_base in r_base_vals:
        for ratio in r_ratio_vals:
            for theta in theta_vals:
                r1 = r_base * ratio
                r2 = r_base
                candidates.append((r1, r2, theta))
                # Also try swapped
                candidates.append((r2, r1, theta))
    
    # Also try some random samples
    np.random.seed(42)
    for _ in range(200):
        r1 = np.random.uniform(0.2, 2.0)
        r2 = np.random.uniform(0.2, 2.0)
        theta = np.random.uniform(0, np.pi/4)
        candidates.append((r1, r2, theta))
    
    # Evaluate all candidates and keep top 20 for local refinement
    scored_candidates = []
    for r1, r2, theta in candidates:
        if r1 < 0.01 or r2 < 0.01:
            continue
        points = _generate_two_squares(r1, r2, theta)
        score = _score_from_points(points)
        scored_candidates.append((score, (r1, r2, theta)))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 20 for local refinement
    top_candidates = [c[1] for c in scored_candidates[:20]]
    
    # Local refinement from each top candidate
    for initial in top_candidates:
        initial = np.array(initial)
        try:
            result = minimize(
                _negative_score,
                initial,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000, 'maxfev': 5000}
            )
            if result.fun < -best_score:
                best_params = result.x
                best_score = -result.fun
        except:
            pass
    
    # If no valid params found, use regular octagon
    if best_params is None:
        best_params = (1.0, 1.0, np.pi/8)
        best_score = 1.0
    
    # Generate final points
    r1, r2, theta = best_params
    points = _generate_two_squares(r1, r2, theta)
    
    # Ensure distinct points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                points[j][0] += 1e-8
    
    # Convert to list of lists of floats
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END
