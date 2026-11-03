"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Differential Evolution for global search + L-BFGS-B on soft-min surrogate for local polish.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13

# EVOLVE-BLOCK-START

def _compute_score_and_worst(pts):
    """
    Calculate the score and identify the worst triangle.
    Returns (score, worst_indices) where worst_indices is a tuple (i, j, k) or None.
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0, None
        
    if hull_area < 1e-10:
        return 0.0, None

    n = len(pts)
    min_area = float('inf')
    worst_idx = None
    
    # Vectorized computation for speed
    # Pre-compute indices for all unique triples
    # To save memory/time, we can just iterate if N is small, but vectorized is fine for N=13
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    worst_idx = (i[min_idx], j[min_idx], k[min_idx])
    
    score = min_area / hull_area
    return score, worst_idx

def _soft_min_objective(x, sharpness=50.0):
    """
    Differentiable objective function for optimization.
    We maximize the minimum area. To make it differentiable, we use a soft-min approximation.
    However, standard L-BFGS-B minimizes. So we minimize -soft_min.
    
    The soft min of areas A_i is: -1/beta * log(sum(exp(-beta * A_i)))
    As beta -> inf, this approaches min(A_i).
    
    We also need to handle the hull area normalization.
    Hull area is not differentiable everywhere, but it's piecewise linear (convex function).
    Actually, area of convex hull is a convex function of the points.
    So min_area / hull_area is a ratio.
    
    A simpler robust approach for the surrogate:
    Objective to minimize: -(min_area / hull_area) is non-smooth.
    
    Let's use the soft-min on the areas directly, and normalize by a fixed scale or the current hull area?
    If we normalize by hull area inside the differentiable function, it becomes complex because hull area calculation is not easily differentiable via standard autograd in scipy without custom gradients.
    
    Alternative: 
    The problem is scale invariant. We can fix the scale of the hull area to 1 during optimization, or just optimize the raw minimum area and normalize at the end? 
    No, because moving points changes the hull area.
    
    Let's try a different surrogate:
    f(x) = - (soft_min(areas) / hull_area)
    Since hull_area is convex and positive, and soft_min is smooth, the ratio is smooth except where hull_area=0 (which we avoid).
    However, scipy's minimize requires numerical gradients or analytic ones.
    Numerical gradients of the ratio involving ConvexHull might be unstable or slow.
    
    Better Strategy:
    Use a "hard" min for the DE evaluation (it's robust to non-smoothness).
    For the local polish, use a smooth approximation that correlates with the hard min.
    
    Let's define the smooth objective as:
    J(x) = - (1 / hull_area) * soft_min(areas)
    
    To compute gradients efficiently, we might need to implement them or rely on finite differences.
    Finite differences for 26 variables are 52 evaluations per step.
    With L-BFGS-B, this is feasible if we limit iterations.
    
    However, ConvexHull is expensive.
    Let's try to avoid ConvexHull in the inner loop of the gradient calculation if possible?
    Or just accept the cost. N=13 is small.
    
    Let's implement a helper to get areas and hull area quickly.
    """
    pts = x.reshape(N, 2)
    
    # Compute areas
    i, j, k = np.indices((N, N, N))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    beta = sharpness
    # Numerical stability: subtract max
    # We want min, so we use -max of -areas
    # log-sum-exp trick
    max_neg_area = np.max(-areas)
    log_sum_exp = max_neg_area + np.log(np.sum(np.exp(-areas - max_neg_area)))
    soft_min_val = -log_sum_exp / beta
    
    # Hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10 # Large penalty
    
    if hull_area < 1e-10:
        return 1e10
        
    score = soft_min_val / hull_area
    return -score # We want to maximize score, so minimize -score

def _local_polish(pts, time_limit=2.0):
    """
    Apply L-BFGS-B to the soft-min objective starting from pts.
    Returns improved points and score.
    """
    x0 = pts.flatten()
    start_time = time.time()
    
    # Use a higher sharpness for better approximation to hard min
    sharpness = 100.0
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10 # Force termination
        return _soft_min_objective(x, sharpness)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 50, 'maxfun': 200})
        
        # Check if we improved
        new_pts = res.x.reshape(N, 2)
        new_score, _ = _compute_score_and_worst(new_pts)
        old_score, _ = _compute_score_and_worst(pts)
        
        if new_score > old_score:
            return new_pts, new_score
        else:
            return pts, old_score
    except Exception:
        return pts, _compute_score_and_worst(pts)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate diverse initial points
    # Use Halton sequences for quasi-random distribution
    def halton_sequence(n, base):
        seq = np.zeros((n, 2))
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += rem * f
            seq[i-1, 0] = r
            # For y coordinate, use next base or different logic? 
            # Standard 2D Halton uses base 2 for x, base 3 for y
            f = 1.0
            r = 0.0
            i_val = i
            base_y = 3 if base == 2 else 2
            while i_val > 0:
                i_val, rem = divmod(i_val, base_y)
                f /= base_y
                r += rem * f
            seq[i-1, 1] = r
        return seq

    # Initialize a set of starting points
    starts = []
    
    # Halton 2
    pts_h = halton_sequence(N, 2)
    # Normalize to unit area roughly
    try:
        hull = ConvexHull(pts_h)
        area = hull.volume
        if area > 1e-10:
            pts_h /= np.sqrt(area)
    except:
        pass
    starts.append(pts_h)
    
    # Halton 3
    pts_h3 = halton_sequence(N, 3)
    try:
        hull = ConvexHull(pts_h3)
        area = hull.volume
        if area > 1e-10:
            pts_h3 /= np.sqrt(area)
    except:
        pass
    starts.append(pts_h3)
    
    # Random
    rng = np.random.default_rng(42)
    pts_r = rng.normal(0, 1, (N, 2))
    try:
        hull = ConvexHull(pts_r)
        area = hull.volume
        if area > 1e-10:
            pts_r /= np.sqrt(area)
    except:
        pass
    starts.append(pts_r)
    
    # Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_p = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(pts_p)
    
    best_pts = None
    best_score = -1.0
    
    # 2. Differential Evolution Global Search
    # Bounds: points should be in a reasonable range, e.g., [-5, 5]
    bounds = [(-5, 5)] * (2 * N)
    
    # DE requires a population. 
    # We will run DE for a fixed number of generations or time.
    # Since DE can be slow, we use a moderate population size.
    
    # We'll run DE on the raw score function (hard min)
    def de_objective(x):
        if time.time() > deadline - 10:
            return 1e10
        pts = x.reshape(N, 2)
        score, _ = _compute_score_and_worst(pts)
        return -score # Minimize negative score
    
    try:
        # maxiter is number of generations. 
        # popsize=15 means population size is 15 * (2*N) = 15*26 = 390 points? 
        # No, popsize is a factor of the dimension.
        # Let's use a fixed popsize for clarity, or let DE decide.
        # For 26 dimensions, popsize=20 is ~520 evaluations per generation.
        # 50 generations = 26000 evaluations.
        # Each eval is fast (vectorized). 
        # Let's try maxiter=30, popsize=15.
        
        # Use the best starting point as a seed? DE doesn't directly take a single seed for the whole run, 
        # but we can initialize the population around the best start if we use a custom init.
        # Standard DE init is uniform in bounds.
        # We can manually set the initial population to include our good starts.
        
        # However, scipy DE doesn't easily allow injecting specific individuals into the initial population 
        # unless we use 'init' as a list of arrays? 
        # Actually, 'init' can be a sequence of arrays if we want to specify the initial population.
        
        # Let's create a custom initial population
        pop = []
        # Add our starts
        for s in starts:
            pop.append(s.flatten())
        
        # Fill the rest with random points in bounds
        target_pop_size = 30 # 30 individuals
        while len(pop) < target_pop_size:
            p = rng.uniform(-5, 5, size=(N, 2))
            # Normalize roughly
            try:
                hull = ConvexHull(p)
                area = hull.volume
                if area > 1e-10:
                    p /= np.sqrt(area)
            except:
                pass
            pop.append(p.flatten())
        
        pop = np.array(pop)
        
        res = differential_evolution(
            de_objective, 
            bounds, 
            init=pop,
            maxiter=50,
            popsize=target_pop_size,
            tol=1e-5,
            polish=False, # We do our own polish
            seed=42,
            update_interval=5,
            callback=None
        )
        
        if res.success or res.fun < -best_score:
            candidate_pts = res.x.reshape(N, 2)
            candidate_score, _ = _compute_score_and_worst(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts
    except Exception as e:
        print(f"DE failed: {e}")
    
    # 3. Local Polish on the best result from DE
    if best_pts is not None:
        polished_pts, polished_score = _local_polish(best_pts, time_limit=20.0)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
            
    # 4. If DE failed or didn't find anything good, fall back to polishing the initial starts
    if best_pts is None or best_score < 1e-6:
        for s in starts:
            if time.time() > deadline - 10:
                break
            polished_pts, polished_score = _local_polish(s, time_limit=5.0)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
    
    # 5. Final check
    if best_pts is None:
        # Absolute fallback
        best_pts = starts[0]
        best_score, _ = _compute_score_and_worst(best_pts)

    return best_pts

# EVOLVE-BLOCK-END
