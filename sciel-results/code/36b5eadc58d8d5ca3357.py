import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 150.0
_START_TIME = time.time()

def _check_time():
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        raise TimeoutError()

def _objective_and_grad(x):
    """
    Objective: Minimize 56 * log(D) - 2 * sum_{i<j} log(d_ij)
    This is equivalent to maximizing the score.
    Returns (value, gradient)
    """
    n = 8
    pts = x.reshape(n, 2)
    
    # Compute pairwise distances
    # d_ij = ||p_i - p_j||
    # We need sum of log(d_ij) for i < j
    # And max d_ij
    
    log_sum = 0.0
    D = 0.0
    max_i, max_j = 0, 1
    dists = np.zeros((n, n))
    
    # Gradient of sum of log distances
    # d/dp_i log(d_ij) = (p_i - p_j) / d_ij^2
    grad_log_sum = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d_sq = np.dot(diff, diff)
            if d_sq < 1e-12:
                # Avoid log(0) and division by zero
                # Return a large penalty
                return 1e10, np.zeros_like(x)
            
            d = np.sqrt(d_sq)
            log_sum += np.log(d)
            
            # Update diameter
            if d > D:
                D = d
                max_i = i
                max_j = j
            
            # Gradient contribution for log(d_ij)
            # d/dp_i log(d) = (p_i - p_j) / d^2
            # d/dp_j log(d) = (p_j - p_i) / d^2
            inv_d_sq = 1.0 / d_sq
            grad_log_sum[i] += diff * inv_d_sq
            grad_log_sum[j] -= diff * inv_d_sq

    if D < 1e-12:
        return 1e10, np.zeros_like(x)
    
    log_D = np.log(D)
    
    # Objective value
    # Score log = 2 * log_sum - 56 * log_D + const
    # Minimize -Score log = 56 * log_D - 2 * log_sum
    val = 56.0 * log_D - 2.0 * log_sum
    
    # Gradient
    # grad_val = 56 * grad(log_D) - 2 * grad(log_sum)
    
    # grad(log_D):
    # D = ||p_max_i - p_max_j||
    # d log_D / dp_max_i = (p_max_i - p_max_j) / D^2
    # d log_D / dp_max_j = (p_max_j - p_max_i) / D^2
    diff_max = pts[max_i] - pts[max_j]
    grad_log_D = np.zeros_like(pts)
    grad_log_D[max_i] = diff_max / (D * D)
    grad_log_D[max_j] = -diff_max / (D * D)
    
    grad = 56.0 * grad_log_D.flatten() - 2.0 * grad_log_sum.flatten()
    
    return val, grad

def _objective_only(x):
    """
    Objective for differential evolution (no gradient needed).
    """
    val, _ = _objective_and_grad(x)
    return val

def _refine(x0):
    """
    Polish the solution using L-BFGS-B or BFGS.
    """
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-10}
        )
        if res.success or res.fun < _objective_only(x0):
            return res.x
    except Exception:
        pass
    return x0

def _compute_score(pts):
    """
    Compute the actual score Delta / n^n.
    """
    n = len(pts)
    pts = np.array(pts, dtype=float)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    dmax = np.max(dists)
    if dmax < 1e-15:
        return 0.0
    
    scaled_dists = dists / (dmax / 2.0)
    mask = ~np.eye(n, dtype=bool)
    prod_dists = scaled_dists[mask]
    
    log_delta = np.sum(np.log(prod_dists))
    score = np.exp(log_delta - n * np.log(n))
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n = 8
    bounds = [(-10, 10)] * (2 * n)
    
    best_x = None
    best_score = -np.inf
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    x_oct = oct_pts.flatten()
    score_oct = _compute_score(oct_pts)
    best_x = x_oct
    best_score = score_oct
    
    # 2. Differential Evolution
    # DE is stochastic. We run it if we have time.
    # popsize=10, maxiter=50. 10*16=160 individuals. 50*160=8000 evals.
    # Each eval is ~50 flops * 8^2 ~ 3000 ops. 8000 * 3000 = 24M ops. < 1s.
    # We can afford more.
    
    try:
        # Run DE
        de_result = differential_evolution(
            _objective_only,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-10,
            init='sobol'  # Better coverage
        )
        
        if de_result.fun < _objective_only(best_x):
            candidate = de_result.x
            # Refine
            candidate = _refine(candidate)
            cand_score = _compute_score(candidate.reshape(n, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_x = candidate
    except Exception:
        pass
    
    # 3. Multi-start Local Optimization with diverse initial guesses
    # If DE didn't find something great, try specific structural guesses.
    
    inits = []
    
    # Regular octagon
    inits.append(x_oct)
    
    # Two squares (concentric)
    # Outer square radius 1, inner square radius r, rotated 45 deg
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        inner_angles = np.pi/4 + np.linspace(0, 2*np.pi, 4, endpoint=False)
        inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        pts = np.vstack([outer, inner])
        inits.append(pts.flatten())
        
    # Grid 2x4
    # Points at (-1, -1), (-1, 1), (1, -1), (1, 1) and 4 more?
    # Let's try a rectangular grid 2x4
    # x in {-1, 1}, y in {-1.5, -0.5, 0.5, 1.5}
    x_coords = [-1, 1]
    y_coords = [-1.5, -0.5, 0.5, 1.5]
    grid_pts = []
    for x in x_coords:
        for y in y_coords:
            grid_pts.append([x, y])
    inits.append(np.array(grid_pts).flatten())
    
    # Random inits
    for _ in range(20):
        pts = np.random.uniform(-1, 1, (8, 2))
        inits.append(pts.flatten())
        
    # Perturbed octagon
    for _ in range(10):
        pts = oct_pts + np.random.normal(0, 0.2, (8, 2))
        inits.append(pts.flatten())
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        # Refine each init
        try:
            x_ref = _refine(x0)
            score = _compute_score(x_ref.reshape(n, 2))
            if score > best_score:
                best_score = score
                best_x = x_ref
        except Exception:
            continue
            
    # Final result
    result_pts = best_x.reshape(n, 2)
    
    # Ensure distinctness and bounds
    # The problem states |coordinate| < 10^6. Our coords are ~1.
    # Distinctness: if two points are too close, perturb.
    # This is unlikely with optimization, but just in case.
    
    result = []
    for p in result_pts:
        x, y = float(p[0]), float(p[1])
        x = max(-1e6, min(1e6, x))
        y = max(-1e6, min(1e6, y))
        result.append([x, y])
        
    # Check distinctness
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = math.sqrt((result[i][0]-result[j][0])**2 + (result[i][1]-result[j][1])**2)
            if d < 1e-6:
                result[j][0] += 1e-4
                result[j][1] += 1e-4
                
    return result

import math
