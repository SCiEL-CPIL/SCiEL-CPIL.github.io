"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150s kill

def _compute_score(pts):
    """Compute Delta / n^n for a set of n points.
    
    Delta = product over all ordered pairs (i,j), i != j of |pi - pj|,
    after scaling the set so its diameter is 2.
    """
    n = len(pts)
    pts = np.array(pts, dtype=float)
    
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # Diameter
    dmax = np.max(dists)
    if dmax < 1e-15:
        return 0.0
    
    # Scale so diameter = 2
    scaled_dists = dists / (dmax / 2.0)
    
    # Product over ordered pairs (i != j)
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    prod_dists = scaled_dists[mask]
    
    # Delta = product of all these distances
    # Use log to avoid underflow/overflow
    log_delta = np.sum(np.log(prod_dists))
    
    # Score = Delta / n^n = exp(log_delta) / n^n
    score = math.exp(log_delta - n * math.log(n))
    
    return score

def _log_product(pts):
    """Compute log of product of pairwise distances (i<j) for optimization.
    We maximize this, subject to diameter constraint handled via penalty.
    """
    n = len(pts)
    pts = np.array(pts, dtype=float)
    
    # All pairwise distances for i < j
    total_log = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return -1e10  # very bad
            total_log += math.log(d)
    
    # The score involves product over ordered pairs, so it's (product over i<j d_ij)^2
    # After scaling to diameter 2: each distance becomes d_ij / (diam/2)
    # So log of scaled product = 2 * sum_{i<j} log(d_ij) - 2*C(n,2)*log(diam/2)
    # And score = exp(log_scaled_product) / n^n
    # But for optimization, we can just maximize the unscaled log product
    # and use diameter as a penalty/constraint.
    
    return total_log

def _diameter(pts):
    """Compute the diameter (max pairwise distance)."""
    n = len(pts)
    pts = np.array(pts, dtype=float)
    max_d = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            d = max(d, max_d)
    return max_d

def _objective(pts_flat):
    """Objective to minimize: negative log product of pairwise distances
    plus a penalty if diameter > 2 (or we fix diameter = 2 via scaling).
    
    Actually, since the score is invariant to scaling (we scale to diameter 2),
    we can just maximize the product of pairwise distances.
    The scaling doesn't change the relative ordering.
    
    Wait, let me re-derive. Score = Delta/n^n where Delta is product over ordered
    pairs after scaling to diameter 2.
    
    If we have points with diameter D, after scaling to diameter 2, each distance
    is multiplied by 2/D. There are n(n-1) ordered pairs.
    
    Delta_scaled = product_{i!=j} (d_ij * 2/D) = (2/D)^{n(n-1)} * product_{i!=j} d_ij
                 = (2/D)^{n(n-1)} * (product_{i<j} d_ij)^2
    
    Score = Delta_scaled / n^n
    
    To maximize score, we maximize log(Delta_scaled) - n*log(n)
    = n(n-1)*log(2) - n(n-1)*log(D) + 2*sum_{i<j} log(d_ij) - n*log(n)
    = constant + 2*sum_{i<j} log(d_ij) - n(n-1)*log(D)
    
    So we want to maximize: 2*sum_{i<j} log(d_ij) - n(n-1)*log(D)
    
    Equivalently, minimize: n(n-1)*log(D) - 2*sum_{i<j} log(d_ij)
    """
    n = 8
    pts = pts_flat.reshape(n, 2)
    
    # Compute sum of log distances for i < j
    log_prod = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e10
            log_prod += math.log(d)
    
    # Compute diameter
    D = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            D = max(D, d)
    
    if D < 1e-15:
        return 1e10
    
    # Objective to minimize
    obj = n * (n-1) * math.log(D) - 2.0 * log_prod
    
    return obj

def _objective_grad(pts_flat):
    """Gradient of the objective function."""
    n = 8
    pts = pts_flat.reshape(n, 2)
    
    # Sum of log distances
    log_prod = 0.0
    D = 0.0
    max_pair = (0, 1)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return np.zeros_like(pts_flat)
            log_prod += math.log(d)
            if d > D:
                D = d
                max_pair = (i, j)
    
    if D < 1e-15:
        return np.zeros_like(pts_flat)
    
    grad = np.zeros_like(pts_flat)
    
    # Gradient of -2 * sum_{i<j} log(d_ij)
    for i in range(n):
        for j in range(i+1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/d(pts[i]) log(d) = (pts[i] - pts[j]) / d^2
            # d/d(pts[j]) log(d) = (pts[j] - pts[i]) / d^2
            grad[2*i:2*i+2] += -2.0 * diff / (d * d)
            grad[2*j:2*j+2] += 2.0 * diff / (d * d)
    
    # Gradient of n(n-1) * log(D)
    # D is the max distance, so gradient only from the max pair
    i, j = max_pair
    diff = pts[i] - pts[j]
    d = np.linalg.norm(diff)
    if d > 1e-15:
        coeff = n * (n-1) / d
        grad[2*i:2*i+2] += coeff * diff / d
        grad[2*j:2*j+2] -= coeff * diff / d
    
    return grad

def _regular_octagon():
    """Return regular octagon points."""
    pts = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        pts.append([math.cos(angle), math.sin(angle)])
    return np.array(pts, dtype=float)

def _concentric_squares():
    """Return two concentric squares (4+4 points) as an initial guess."""
    pts = []
    # Inner square
    r1 = 0.5
    for k in range(4):
        angle = 2 * math.pi * k / 4 + math.pi / 4
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle)])
    # Outer square
    r2 = 1.0
    for k in range(4):
        angle = 2 * math.pi * k / 4
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle)])
    return np.array(pts, dtype=float)

def _random_init(scale=1.0):
    """Random initialization within a disk of given scale."""
    pts = []
    for _ in range(8):
        r = scale * math.sqrt(random.random())
        theta = 2 * math.pi * random.random()
        pts.append([r * math.cos(theta), r * math.sin(theta)])
    return np.array(pts, dtype=float)

def _refine(pts_init, time_limit):
    """Refine a set of points using BFGS optimization."""
    n = 8
    x0 = pts_init.flatten()
    
    start_time = time.time()
    
    def time_callback():
        if time.time() - start_time > time_limit:
            return True
        return None
    
    result = minimize(
        _objective,
        x0,
        jac=_objective_grad,
        method='BFGS',
        options={
            'maxiter': 500,
            'gtol': 1e-10,
            'callback': time_callback
        }
    )
    
    if result.success or result.fun < _objective(x0):
        best_pts = result.x.reshape(n, 2)
        best_obj = result.fun
    else:
        best_pts = pts_init
        best_obj = _objective(x0)
    
    return best_pts, best_obj

def _score_from_points(pts):
    """Compute the actual score from points."""
    return _compute_score(pts)

def construct():
    """Main entry point: find 8 points maximizing the product of distances
    at fixed diameter."""
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    best_score = 0.0
    best_points = None
    
    # Multiple initializations to escape local optima
    inits = []
    
    # 1. Regular octagon
    inits.append(_regular_octagon())
    
    # 2. Concentric squares with various ratios
    for r1 in [0.3, 0.5, 0.7, 0.85]:
        for offset in [0, math.pi/4, math.pi/8, 3*math.pi/8]:
            pts = []
            for k in range(4):
                angle = 2 * math.pi * k / 4 + offset
                pts.append([r1 * math.cos(angle), r1 * math.sin(angle)])
            for k in range(4):
                angle = 2 * math.pi * k / 4
                pts.append([math.cos(angle), math.sin(angle)])
            inits.append(np.array(pts, dtype=float))
    
    # 3. Two concentric squares with different rotations
    for r1 in [0.4, 0.6, 0.8]:
        for rot in [math.pi/8, math.pi/6, math.pi/5, math.pi/3]:
            pts = []
            for k in range(4):
                angle = 2 * math.pi * k / 4 + rot
                pts.append([r1 * math.cos(angle), r1 * math.sin(angle)])
            for k in range(4):
                angle = 2 * math.pi * k / 4
                pts.append([math.cos(angle), math.sin(angle)])
            inits.append(np.array(pts, dtype=float))
    
    # 4. Random inits
    for _ in range(10):
        inits.append(_random_init(scale=1.0))
    
    # 5. Perturbed regular octagon
    base = _regular_octagon()
    for _ in range(5):
        perturbed = base + 0.1 * np.random.randn(8, 2)
        inits.append(perturbed)
    
    # Try each initialization
    for i, init in enumerate(inits):
        if time.time() >= deadline:
            break
        
        # Time limit per refinement: use remaining time divided by remaining inits
        remaining_inits = len(inits) - i
        time_per_init = (deadline - time.time()) / max(1, remaining_inits)
        time_per_init = max(1.0, min(time_per_init, 10.0))  # at least 1s, at most 10s
        
        try:
            refined_pts, obj = _refine(init, time_per_init)
            score = _score_from_points(refined_pts)
            
            if score > best_score:
                best_score = score
                best_points = refined_pts.copy()
        except Exception:
            continue
    
    # If we still have time, do additional random restarts
    if best_points is None:
        # Fallback to regular octagon
        best_points = _regular_octagon()
        best_score = _score_from_points(best_points)
    
    # Additional refinement on the best point found
    if time.time() < deadline - 5.0:
        try:
            refined_pts, obj = _refine(best_points, min(20.0, deadline - time.time() - 2.0))
            score = _score_from_points(refined_pts)
            if score > best_score:
                best_score = score
                best_points = refined_pts.copy()
        except Exception:
            pass
    
    # Ensure points are distinct and coordinates are bounded
    result = []
    for p in best_points:
        x, y = float(p[0]), float(p[1])
        # Clamp to avoid huge values (shouldn't happen but safety)
        x = max(-1e6, min(1e6, x))
        y = max(-1e6, min(1e6, y))
        result.append([x, y])
    
    # Ensure distinctness
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = math.sqrt((result[i][0]-result[j][0])**2 + (result[i][1]-result[j][1])**2)
            if d < 1e-6:
                # Separate them slightly
                result[j][0] += 1e-4
                result[j][1] += 1e-4
    
    return result
# EVOLVE-BLOCK-END
