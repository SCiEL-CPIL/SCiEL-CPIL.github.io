"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings("ignore")

DEADLINE = 150.0
START_TIME = time.time()

def _remaining():
    return DEADLINE - (time.time() - START_TIME)

def _score(pts):
    """Compute the radius: min over pairs and boundaries."""
    # Boundary distance
    boundary = 0.5 - np.max(np.abs(pts), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    d = pdist(pts)
    min_pair = np.min(d)
    return min(min_boundary, min_pair / 2.0)

def _neg_score(pts):
    return -_score(pts)

def _soft_neg_score(pts, t=1.0):
    """Soft objective using log-sum-exp for smoothness."""
    # Boundary
    boundary = 0.5 - np.max(np.abs(pts), axis=1)
    # Pairwise
    d = pdist(pts)
    # We want to maximize min(min_boundary, min_d/2)
    # Equivalent to maximizing t * log( (sum exp(-min_boundary/t))^-1 ... )
    # Actually, let's use a simpler soft-min:
    # soft_min(x, t) = -t * log(sum(exp(-x_i/t)))
    # We want max(soft_min(boundary, t), soft_min(d/2, t))
    # But the objective is min(boundary, min_d/2).
    # Let's just use the hard min but with a smoothing parameter if needed.
    # For now, let's stick to the hard min for the main optimization, 
    # but maybe use a barrier for boundaries.
    
    # Let's use a penalty for boundary violations to help the optimizer.
    # Or just use the hard score.
    
    # Let's try a different approach: maximize r such that all constraints are satisfied.
    # This is an epigraph formulation.
    # Maximize r
    # Subject to:
    #   0.5 - max|c_i| >= r  =>  c_i in [-0.5+r, 0.5-r]^2
    #   ||c_i - c_j|| >= 2r  =>  (c_i-c_j)^2 - (2r)^2 >= 0
    
    # This is non-convex.
    
    # Let's stick to maximizing the score directly with L-BFGS-B.
    # To help L-BFGS-B, we can use a smooth approximation of the min function.
    # soft_min(x, y, t) = -t * log( exp(-x/t) + exp(-y/t) )
    # Here x = boundary, y = min_d/2.
    # But min_d is not smooth.
    
    # Let's just use the hard score. It works well enough with good initializations.
    
    return _neg_score(pts)

def _hexagonal_init():
    """Create a hexagonal packing initialization."""
    # For N=200, a hexagonal grid is a good start.
    # Radius approx 0.0366.
    # Side of hexagon unit cell is 2r.
    # Let's generate points in a hexagonal lattice and clip to the square.
    
    # Estimate r
    # Area = N * pi * r^2 <= 1
    # r <= 1/sqrt(N*pi) ~ 1/sqrt(628) ~ 0.0399
    # Best known is 0.0366.
    
    # Let's start with a grid of spacing 2*0.036 = 0.072
    # Hexagonal spacing:
    # dx = 2r
    # dy = 2r * sqrt(3)/2 = r * sqrt(3)
    
    r_guess = 0.036
    dx = 2 * r_guess
    dy = r_guess * np.sqrt(3)
    
    points = []
    # Iterate over rows
    # Square is [-0.5, 0.5]
    # y ranges from -0.5 to 0.5
    y = -0.5 + dy/2
    row = 0
    while y <= 0.5:
        x_offset = (row % 2) * dx/2
        x = -0.5 + dx/2 + x_offset
        while x <= 0.5:
            # Check if inside
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            x += dx
        y += dy
        row += 1
    
    points = np.array(points)
    
    # If we have too many, keep the first N (or center them?)
    # If we have too few, add more?
    # Hexagonal grid might have more or less than N.
    # Let's just take the first N if len(points) >= N.
    # If len(points) < N, we might need to adjust.
    
    # For N=200, let's see how many points we get.
    # dx=0.072, dy=0.062
    # x count ~ 1/0.072 ~ 14
    # y count ~ 1/0.062 ~ 16
    # Total ~ 14*16 = 224.
    # So we should have enough.
    
    if len(points) >= N:
        # Center the set? Or just take the first N?
        # Taking the first N might bias towards the top-left.
        # Let's sort by distance from center to keep the densest part?
        # Or just take a random subset?
        # Let's take the points closest to the center to ensure they are well inside.
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:N]
        return points[idx]
    else:
        # Fallback to square grid if hexagonal is too sparse
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

def _random_init():
    """Random initialization inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9

def _optimize(pts, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B optimization."""
    if _remaining() < 5:
        return pts
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        _neg_score,
        pts,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
    )
    
    if _remaining() < 2:
        return pts
    
    if result.success or result.fun < _neg_score(pts):
        return result.x
    else:
        return pts

def construct():
    # Try multiple initializations and keep the best.
    best_pts = None
    best_score = -1.0
    
    # 1. Hexagonal init
    pts_hex = _hexagonal_init()
    score_hex = _score(pts_hex)
    if score_hex > best_score:
        best_score = score_hex
        best_pts = pts_hex
    
    # Optimize from hexagonal
    if _remaining() > 20:
        pts_opt = _optimize(pts_hex, max_iter=1000, ftol=1e-12, gtol=1e-10)
        score_opt = _score(pts_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # 2. Square grid init
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts_grid = grid[:N]
    score_grid = _score(pts_grid)
    if score_grid > best_score:
        best_score = score_grid
        best_pts = pts_grid
    
    # Optimize from grid
    if _remaining() > 20:
        pts_opt = _optimize(pts_grid, max_iter=1000, ftol=1e-12, gtol=1e-10)
        score_opt = _score(pts_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # 3. Random inits
    num_random = 10
    for i in range(num_random):
        if _remaining() < 10:
            break
        pts_rand = _random_init()
        score_rand = _score(pts_rand)
        if score_rand > best_score:
            best_score = score_rand
            best_pts = pts_rand
        
        # Optimize from random
        if _remaining() > 10:
            pts_opt = _optimize(pts_rand, max_iter=500, ftol=1e-12, gtol=1e-10)
            score_opt = _score(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
    
    # 4. Final polish with tighter tolerances
    if _remaining() > 5 and best_pts is not None:
        pts_polish = _optimize(best_pts, max_iter=2000, ftol=1e-15, gtol=1e-12)
        score_polish = _score(pts_polish)
        if score_polish > best_score:
            best_score = score_polish
            best_pts = pts_polish
    
    # Ensure valid output
    if best_pts is None:
        return _hexagonal_init()
    
    # Clip to bounds just in case
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
