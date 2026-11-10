import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize, differential_evolution

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def soft_objective(params, beta):
    """
    Smooth surrogate for -log((d_min/d_max)^2) = 2*log(d_max) - 2*log(d_min).
    We want to minimize this, which is equivalent to maximizing log((d_min/d_max)^2).
    Using log-sum-exp for stability.
    """
    n = 16
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(14, 2)
    
    if not np.all(np.isfinite(pts)):
        return 1e6
        
    dists = spd.pdist(pts)
    if len(dists) == 0:
        return 1e6
        
    # Soft max
    # d_max_soft = beta * log(sum(exp(d/beta)))
    # d_min_soft = -beta * log(sum(exp(-d/beta)))
    
    # For numerical stability, we can just use the raw log-sum-exp if beta is small enough
    # or use the standard stable formulation.
    
    # Objective: minimize 2*log(d_max_soft) - 2*log(d_min_soft)
    # = 2 * (log(beta) + log_sum_exp(d/beta) - (log(-beta) + log_sum_exp(-d/beta)))
    # Note: log(-beta) is constant if beta>0? No, beta is positive. 
    # d_min_soft = -beta * LSE(-d/beta). Since d>0, -d/beta < 0. LSE is positive.
    # Wait, if d are positive, -d/beta are negative. 
    # LSE(x) = log(sum(exp(x))).
    # d_min_soft = -beta * log(sum(exp(-d/beta))).
    # If beta is small, sum(exp(-d/beta)) is dominated by the smallest d (largest -d/beta).
    # Let d_min be the true min. -d_min/beta is the largest value in -d/beta.
    # sum(exp(-d/beta)) ~ exp(-d_min/beta) * (1 + small terms).
    # log(sum(...)) ~ -d_min/beta.
    # d_min_soft ~ -beta * (-d_min/beta) = d_min. Correct.
    
    # However, we must ensure d_min_soft > 0 for log.
    # Since d > 0, -d/beta < 0. exp(-d/beta) < 1. Sum < N. 
    # log(sum) < log(N).
    # If N=120, log(120) ~ 4.8.
    # If beta is small, LSE is negative (large magnitude).
    # d_min_soft = -beta * (negative number) = positive.
    
    # Let's compute the objective value directly.
    # val = 2 * (log(d_max_soft) - log(d_min_soft))
    
    # Stable log-sum-exp
    def lse(x):
        m = np.max(x)
        if not np.isfinite(m):
            return -np.inf
        return m + np.log(np.sum(np.exp(x - m)))
    
    x_max = dists / beta
    x_min = -dists / beta
    
    lse_max = lse(x_max)
    lse_min = lse(x_min)
    
    # log(d_max_soft) = log(beta) + lse_max
    # log(d_min_soft) = log(-beta) + lse_min  -- Wait.
    # d_min_soft = -beta * exp(lse_min) is NOT correct.
    # d_min_soft = -beta * log(sum(exp(-d/beta))) is the definition used in some contexts?
    # No, the standard soft min is:
    # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # Here we want soft min of distances d.
    # soft_min_d = -1/beta * log(sum(exp(-beta*d)))
    # soft_max_d = 1/beta * log(sum(exp(beta*d)))
    
    # Let's use this standard definition.
    # soft_max_d = (1/beta) * lse(beta*d)
    # soft_min_d = -(1/beta) * lse(-beta*d)
    
    # We want to maximize (d_min / d_max)^2.
    # Equivalent to maximizing 2*log(d_min) - 2*log(d_max).
    # So minimize 2*log(d_max) - 2*log(d_min).
    
    # log(soft_max_d) = log(1/beta) + lse(beta*d)
    # log(soft_min_d) = log(1/beta) + lse(-beta*d)  -- Wait.
    # soft_min_d is positive.
    # log(soft_min_d) = log(-(1/beta) * lse(-beta*d))? No.
    # soft_min_d = -1/beta * log(sum(exp(-beta*d))).
    # Let L = log(sum(exp(-beta*d))).
    # soft_min_d = -1/beta * exp(L).
    # log(soft_min_d) = -log(beta) + L.
    
    # So objective to minimize:
    # 2 * (log(soft_max_d) - log(soft_min_d))
    # = 2 * ( (-log(beta) + lse(beta*d)) - (-log(beta) + lse(-beta*d)) )
    # = 2 * ( lse(beta*d) - lse(-beta*d) )
    
    # The log(beta) terms cancel out! This is nice.
    
    val = 2.0 * (lse_max - lse_min)
    
    if not np.isfinite(val):
        return 1e6
        
    return val

def exact_objective(params):
    """
    Exact objective for polishing.
    Minimizes -calculate_score(params)
    """
    n = 16
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(14, 2)
    
    if not np.all(np.isfinite(pts)):
        return 1e6
        
    score = calculate_score(pts)
    return -score

def params_to_points(params):
    n = 16
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(14, 2)
    return pts

def points_to_params(pts):
    """
    Convert 16x2 points to 28-dim params.
    We need to identify which two points are the "fixed" ones.
    For initialization, we just pick the two closest points, translate/rotate/scale them to (0,0) and (1,0),
    and extract the rest.
    """
    dists = spd.pdist(pts)
    if len(dists) == 0:
        return None
    min_idx = np.argmin(dists)
    
    # Find pair (i, j)
    idx = 0
    i_found = j_found = -1
    for i in range(15):
        for j in range(i+1, 16):
            if idx == min_idx:
                i_found, j_found = i, j
                break
            idx += 1
        if i_found >= 0:
            break
    
    if i_found < 0:
        return None
        
    # Reorder so i_found is 0 and j_found is 1
    # This is a heuristic initialization. The optimizer doesn't care about indices,
    # only the set of points.
    # We just need *some* valid parameter vector that represents the configuration.
    # Since the objective is symmetric with respect to permutation of points 2..15,
    # any permutation of the free points yields the same score.
    # So we can just take pts[0] and pts[1] as the reference?
    # No, pts[0] and pts[1] in the input array might not be the closest pair.
    # But we can just construct the params from the array as is, assuming the first two are the reference.
    # Wait, if we do that, the "fixed" points in the parameter space are the first two in the array.
    # The optimizer will move points 2..15.
    # If the global optimum has a different pair as the "closest", the optimizer might struggle
    # if the first two points in our initial guess are not the closest.
    # However, Differential Evolution explores the space of the 14 free points.
    # It can move points 2..15 anywhere.
    # It cannot move points 0 and 1.
    # So, if the true optimal configuration has its closest pair NOT at indices 0 and 1,
    # and we fix indices 0 and 1 to be the closest pair (distance 1), we are restricting the search.
    # BUT, the problem is symmetric. Any pair can be the closest.
    # So, if we fix ANY pair to be the closest (distance 1), we cover the symmetry class.
    # We just need to ensure that in our initial guess, the pair at (0,0) and (1,0) IS the closest pair.
    # If it's not, the "distance 1" constraint is wrong for that configuration.
    
    # Strategy:
    # 1. Find the closest pair in the input pts.
    # 2. Rotate/translate/scale so that this pair is at (0,0) and (1,0).
    # 3. The other 14 points become the params.
    
    p0 = pts[i_found].copy()
    p1 = pts[j_found].copy()
    
    # Translate
    pts_t = pts - p0
    # p0 is now at 0,0. p1 is at v = p1 - p0.
    # Scale
    d = np.linalg.norm(pts_t[j_found])
    if d < 1e-9:
        return None
    pts_s = pts_t / d
    # Now pts_s[0] is 0,0? No, pts_s[i_found] is 0,0.
    # We want index 0 to be 0,0 and index 1 to be 1,0.
    # So we need to permute the array.
    
    # Create a new array where i_found is at index 0 and j_found is at index 1.
    new_pts = np.zeros_like(pts_s)
    new_pts[0] = pts_s[i_found]
    new_pts[1] = pts_s[j_found]
    
    remaining_indices = [k for k in range(16) if k != i_found and k != j_found]
    for idx, k in enumerate(remaining_indices):
        new_pts[2 + idx] = pts_s[k]
        
    # Now new_pts[0] is (0,0) and new_pts[1] is (1,0) (up to rotation).
    # We need to rotate so new_pts[1] is (1,0).
    # new_pts[1] is currently (1,0) because we scaled by its norm and it was the vector from p0 to p1.
    # Wait, p1 - p0 had norm d. Divided by d, it has norm 1.
    # But its direction is arbitrary.
    # We need to rotate the whole system so that new_pts[1] aligns with x-axis.
    
    v = new_pts[1]
    angle = np.arctan2(v[1], v[0])
    c = np.cos(-angle)
    s = np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    
    new_pts_rot = new_pts @ rot.T
    
    # Check
    if not np.allclose(new_pts_rot[0], [0,0], atol=1e-6):
        return None
    if not np.allclose(new_pts_rot[1], [1,0], atol=1e-6):
        return None
        
    params = new_pts_rot[2:].flatten()
    return params

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    global_deadline = time.time() + 580  # 580s to leave 20s margin
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # A. Hexagonal lattice subset
    # 1 center, 6 ring, 9 ring (3x3 grid rotated?)
    # Let's just take a 4x4 grid and perturb
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(grid_points)
    
    # B. Hexagonal pattern
    # 1 center
    # 6 at r=1
    # 9 at r=2 (hexagonal arrangement)
    center = np.array([[0,0]])
    r1 = 1.0
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring1 = np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)])
    
    r2 = 2.0
    # 9 points in a hexagon?
    # 1 at (2,0), 2 at (1, sqrt(3)), 2 at (-1, sqrt(3)), 2 at (-1, -sqrt(3)), 2 at (1, -sqrt(3))?
    # That's 9 points?
    # Let's just use a slightly perturbed grid for simplicity in init, or a known good packing.
    # For N=16, a 4x4 square grid is a decent start.
    # Let's add a "hexagonal" init by taking the 16 closest points to origin in a hex lattice.
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x_h = i + 0.5 * j
            y_h = np.sqrt(3) / 2 * j
            hex_pts.append([x_h, y_h])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    d_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(d_center)[:16]
    hex_init = hex_pts[idx]
    candidates.append(hex_init)
    
    # C. Random
    rng = np.random.RandomState(42)
    for _ in range(3):
        rand_pts = rng.randn(16, 2) * 2.0
        candidates.append(rand_pts)
        
    # Convert to params
    param_inits = []
    for cand in candidates:
        p = points_to_params(cand)
        if p is not None:
            param_inits.append(p)
            
    if not param_inits:
        # Fallback
        param_inits.append(rng.randn(28) * 2.0)
        
    best_params = None
    best_score = 0.0
    
    # 2. Global Search with Differential Evolution
    # Bounds: points shouldn't be too far. If d_max is too large, R drops.
    # If d_min is 0, R is 0.
    # Reasonable bounds: [-5, 5] for each coordinate.
    bounds = [(-5.0, 5.0)] * 28
    
    # Use the soft objective for DE to avoid non-differentiability issues, 
    # although DE doesn't use gradients, the landscape shape matters.
    # A small beta makes it closer to the true objective.
    beta_de = 0.1
    
    def de_objective(params):
        return soft_objective(params, beta_de)
    
    if time.time() < global_deadline - 100:
        try:
            # DE is slow. Limit maxiter.
            # popsize=10, 28 dims.
            # maxiter=100
            de_res = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=100, 
                popsize=15,
                seed=42,
                polish=False, # We will polish separately
                tol=1e-8,
                atol=1e-8
            )
            
            if de_res.success or de_res.fun < 1e5:
                # Check score
                score = calculate_score(params_to_points(de_res.x))
                if score > best_score:
                    best_score = score
                    best_params = de_res.x.copy()
        except Exception as e:
            pass # print(e)
            
    # 3. Local Refinement with L-BFGS-B on Soft Surrogate
    # Use the best params from DE and the inits
    
    # We will run L-BFGS-B on multiple starts: the DE result and the original inits.
    starts = []
    if best_params is not None:
        starts.append(best_params)
    starts.extend(param_inits[:5]) # Top 5 inits
    
    beta_lbfgs = 0.05
    
    for start in starts:
        if time.time() > global_deadline - 50:
            break
            
        try:
            res = minimize(
                soft_objective, 
                start, 
                args=(beta_lbfgs,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < 1e5:
                score = calculate_score(params_to_points(res.x))
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass

    # 4. Final Polish with Nelder-Mead on Exact Objective
    if best_params is not None and time.time() < global_deadline - 10:
        try:
            res = minimize(
                exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Also try polishing the DE result directly if it was better than L-BFGS start?
    # The loop above included DE result in starts.
    
    if best_params is None:
        # Return a valid default if all failed
        final_points = np.column_stack((np.linspace(-2, 2, 16), np.zeros(16)))
    else:
        final_points = params_to_points(best_params)
        
    if not np.all(np.isfinite(final_points)):
        final_points = np.zeros((n, d))
        
    return final_points
