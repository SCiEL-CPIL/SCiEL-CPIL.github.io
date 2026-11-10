# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
import math
from scipy.optimize import basinhopping, minimize

N = 13
# Precompute triangle indices for efficiency
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull
    Returns 0.0 if invalid
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Vectorized area calculation
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate for the objective function.
    We want to maximize A_min / A_hull.
    So we minimize -A_min / A_hull.
    A_min = min(areas).
    Soft min: -T * log(sum(exp(-area/T)))
    """
    if not np.all(np.isfinite(points)):
        return 1e10
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    
    if A_hull <= 0:
        return 1e10
        
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min of areas
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    
    # Objective to minimize: -soft_min_area / A_hull
    return -soft_min_area / A_hull

def generate_initial_candidates(rng):
    """
    Generate diverse initial candidates based on geometric templates.
    """
    candidates = []
    
    # Template 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts)
    
    # Template 2: 1 center + 12 outer (regular 12-gon)
    center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([center, outer])
    candidates.append(pts)
    
    # Template 3: 1 center + 6 inner + 6 outer
    center = np.array([0.0, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([center, inner, outer])
    candidates.append(pts)
    
    # Template 4: 2 center + 5 inner + 6 outer
    c1 = np.array([-0.1, 0.0])
    c2 = np.array([0.1, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([c1, c2, inner, outer])
    candidates.append(pts)
    
    # Template 5: Grid-like
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    pts = grid[:13]
    candidates.append(pts)
    
    # Template 6: Random
    pts = rng.uniform(-1, 1, size=(13, 2))
    candidates.append(pts)
    
    # Template 7: Fibonacci spiral
    golden_angle = 2.39996
    radii = np.sqrt(np.arange(13) / 13.0)
    angles = np.arange(13) * golden_angle
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts)
    
    return candidates

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize 13 points to maximize the minimum triangle area ratio.
    Uses Basin Hopping with L-BFGS-B local optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0 # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial candidates
    candidates = generate_initial_candidates(rng)
    
    # Normalize initial points to unit scale for better optimization behavior
    normalized_candidates = []
    for base in candidates:
        init_pts_centered = base - np.mean(base, axis=0)
        max_norm = np.max(np.linalg.norm(init_pts_centered, axis=1))
        if max_norm > 1e-6:
            init_pts_scaled = init_pts_centered / max_norm
        else:
            init_pts_scaled = init_pts_centered
        normalized_candidates.append(init_pts_scaled.flatten())
    
    # Define objective function for Basin Hopping (minimization)
    # We use a soft-min surrogate. Temperature is fixed for the local optimizer.
    # Basin Hopping will kick the points, allowing escape from local minima.
    
    def objective(x_flat):
        pts = x_flat.reshape(13, 2)
        # Use a slightly higher temperature for the global search to be smoother
        return soft_min_score(pts, temperature=0.05)
        
    def local_optimizer(x_flat):
        # Use L-BFGS-B for fast local convergence
        # We use a lower temperature for the final local polish to be closer to hard min
        # However, basinhopping calls this repeatedly. 
        # It is better to keep the objective consistent or adapt it.
        # Let's use a moderate temperature for the local step.
        res = minimize(
            lambda x: soft_min_score(x.reshape(13, 2), temperature=0.02),
            x_flat,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        return res.x, res.fun, res.nfev
        
    # Run Basin Hopping on the best few candidates
    # We allocate time to run Basin Hopping on the top 3 candidates
    n_runs = 3
    time_per_run = (deadline - start_time) / n_runs
    
    for i, x0 in enumerate(normalized_candidates[:n_runs]):
        current_time = time.time()
        if current_time > deadline:
            break
            
        # Create a time-limited version of the local optimizer?
        # Basinhopping doesn't have a direct time limit, but we can limit niter.
        # Estimating niter: each local step is fast (~ms). 
        # Let's allow up to 200 iterations of basin hopping.
        
        try:
            bh_result = basinhopping(
                objective, 
                x0, 
                minimizer_kwargs=dict(method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6}),
                stepsize=0.5,
                niter=50, # Limit iterations to control time
                seed=42+i,
                interval=10, # Check time every 10 iterations? No, basinhopping doesn't support callback for time break easily without modifying.
                disp=False
            )
            
            pts_opt = bh_result.x.reshape(13, 2)
            
            # Calculate hard score
            score = compute_score(pts_opt)
            
            if score > best_score:
                best_score = score
                best_points = pts_opt.copy()
                
        except Exception as e:
            # If basin hopping fails, fallback to simple local optimization
            try:
                res = minimize(
                    lambda x: soft_min_score(x.reshape(13, 2), temperature=0.02),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                pts_opt = res.x.reshape(13, 2)
                score = compute_score(pts_opt)
                if score > best_score:
                    best_score = score
                    best_points = pts_opt.copy()
            except Exception:
                pass

    # Final Polish: Use a very low temperature to sharpen the minimum
    if best_points is not None:
        polish_deadline = deadline - 5.0
        current_time = time.time()
        if current_time < polish_deadline:
            x0 = best_points.flatten()
            
            def obj_func_polish(x_flat):
                pts = x_flat.reshape(13, 2)
                return soft_min_score(pts, temperature=0.005)
                
            try:
                res = minimize(
                    obj_func_polish,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                pts_polished = res.x.reshape(13, 2)
                
                score = compute_score(pts_polished)
                if score > best_score:
                    best_score = score
                    best_points = pts_polished.copy()
            except Exception:
                pass

    # Fallback
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False))])
        
    return best_points

# EVOLVE-BLOCK-END
