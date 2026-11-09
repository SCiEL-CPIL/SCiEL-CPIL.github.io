"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _compute_score(points):
    """Compute the score: product of all pairwise distances / n^n, after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    # Compute all pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            dists.append(d)
    dists = np.array(dists)
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists * (2.0 / diam)
    # Product of all pairwise distances
    log_prod = np.sum(np.log(scaled_dists))
    prod = np.exp(log_prod)
    score = prod / (n ** n)
    return score

def _neg_log_score(params):
    """Negative log of the score for minimization. params encodes symmetric configuration."""
    # params: [x1, y1, x2, y2, x3, y3, x4, y4] for 4 symmetric pairs
    # Points are: (x1,y1), (x1,-y1), (x2,y2), (x2,-y2), (x3,y3), (x3,-y3), (x4,y4), (x4,-y4)
    # We enforce y_i > 0 by using |y_i| or just letting the optimizer handle it
    n_pairs = 4
    pts = []
    for i in range(n_pairs):
        x = params[2*i]
        y = params[2*i+1]
        pts.append([x, y])
        pts.append([x, -y])
    pts = np.array(pts)
    
    # Compute pairwise distances
    dists = []
    for i in range(8):
        for j in range(i+1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            dists.append(d)
    dists = np.array(dists)
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0  # Degenerate
    scaled_dists = dists * (2.0 / diam)
    log_prod = np.sum(np.log(scaled_dists))
    # Score = exp(log_prod) / 8^8
    # -log(score) = -log_prod + 8*log(8)
    neg_log = -log_prod + 8 * np.log(8.0)
    return neg_log

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with regular octagon as initial guess
    # Regular octagon: vertices at angles 0, 45, 90, ..., 315
    # With x-axis symmetry: points at (1,0), (-1,0) on axis, and 3 pairs
    # Let's use 4-pair parameterization for simplicity
    # Regular octagon in 4-pair form:
    # Angles: 22.5, 67.5, 112.5, 157.5 (and their negatives)
    # Actually let me just use the 4-pair symmetric form
    
    # Regular octagon has vertices at angles 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # In symmetric form about x-axis:
    # (1, 0), (-1, 0) on x-axis
    # (cos(pi/4), sin(pi/4)), (cos(pi/4), -sin(pi/4))
    # (cos(3pi/4), sin(3pi/4)), (cos(3pi/4), -sin(3pi/4))
    # But that's only 2 pairs + 2 on-axis = 6 points... no, 2 pairs = 4 points + 2 = 6. Wrong.
    
    # Let me reconsider. 8 points with x-axis symmetry:
    # If all 4 pairs are off-axis: 8 points, no points on x-axis
    # Regular octagon: if we rotate it so no vertex is on x-axis... 
    # Vertices at angles pi/8, 3pi/8, 5pi/8, 7pi/8, 9pi/8, 11pi/8, 13pi/8, 15pi/8
    # These come in pairs: (pi/8, -pi/8), (3pi/8, -3pi/8), (5pi/8, -5pi/8), (7pi/8, -7pi/8)
    # In x-y: (cos(pi/8), ±sin(pi/8)), (cos(3pi/8), ±sin(3pi/8)), (cos(5pi/8), ±sin(5pi/8)), (cos(7pi/8), ±sin(7pi/8))
    
    init_params = []
    for k in range(4):
        angle = (2*k + 1) * np.pi / 8  # pi/8, 3pi/8, 5pi/8, 7pi/8
        x = np.cos(angle)
        y = np.sin(angle)
        init_params.append(x)
        init_params.append(y)
    init_params = np.array(init_params)
    
    # Verify initial score
    pts_init = []
    for i in range(4):
        x = init_params[2*i]
        y = init_params[2*i+1]
        pts_init.append([x, y])
        pts_init.append([x, -y])
    score_init = _compute_score(pts_init)
    best_params = init_params.copy()
    best_score = score_init
    
    # Define bounds for differential evolution
    # Coordinates should be in a reasonable range, say [-2, 2]
    bounds = [(-2.0, 2.0)] * 8
    
    # Use differential evolution
    try:
        result = differential_evolution(
            _neg_log_score,
            bounds,
            x0=init_params,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,
            init='so-and-poly',
            callback=lambda x, convergence: (time.time() > deadline),
            update_interval=1
        )
        
        if time.time() < deadline:
            # Evaluate the result
            res_params = result.x
            pts_res = []
            for i in range(4):
                x = res_params[2*i]
                y = res_params[2*i+1]
                pts_res.append([x, y])
                pts_res.append([x, -y])
            score_res = _compute_score(pts_res)
            if score_res > best_score:
                best_score = score_res
                best_params = res_params.copy()
    except Exception as e:
        pass
    
    # If we have time, try a few more random restarts
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        np.random.seed(123)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Random perturbation of the best
            perturb = best_params + np.random.randn(8) * 0.5
            # Also try some structured perturbations
            try:
                result = differential_evolution(
                    _neg_log_score,
                    bounds,
                    x0=perturb,
                    seed=100 + restart,
                    maxiter=100,
                    popsize=10,
                    tol=1e-12,
                    mutation=(0.5, 1.5),
                    recombination=0.7,
                    polish=True,
                    init='so-and-poly',
                    callback=lambda x, convergence: (time.time() > deadline - 5),
                    update_interval=1
                )
                if time.time() > deadline - 5:
                    break
                res_params = result.x
                pts_res = []
                for i in range(4):
                    x = res_params[2*i]
                    y = res_params[2*i+1]
                    pts_res.append([x, y])
                    pts_res.append([x, -y])
                score_res = _compute_score(pts_res)
                if score_res > best_score:
                    best_score = score_res
                    best_params = res_params.copy()
            except Exception:
                pass
    
    # Convert best_params to final points
    final_points = []
    for i in range(4):
        x = float(best_params[2*i])
        y = float(best_params[2*i+1])
        final_points.append([x, y])
        final_points.append([x, -y])
    
    # Ensure distinctness
    # Check for near-duplicate points
    unique_points = []
    for p in final_points:
        is_dup = False
        for q in unique_points:
            if abs(p[0] - q[0]) < 1e-10 and abs(p[1] - q[1]) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)
    
    # If we lost points due to duplicates, fall back to regular octagon
    if len(unique_points) < 8:
        return [[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)]
    
    return unique_points[:8]
# EVOLVE-BLOCK-END
