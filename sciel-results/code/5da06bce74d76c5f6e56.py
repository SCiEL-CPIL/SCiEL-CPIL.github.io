import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import sys

N = 14

# Precompute triangle indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    This is differentiable and approximates -min(A) as tau -> 0.
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def generate_ring_points(counts, radii, offsets):
    """
    Generate points from ring definitions.
    counts: list of ints, sum to N
    radii: list of floats
    offsets: list of floats (angles)
    """
    points = []
    for c, r, th in zip(counts, radii, offsets):
        if c > 0:
            angles = th + np.linspace(0, 2*np.pi, c, endpoint=False)
            points.append(np.column_stack((r*np.cos(angles), r*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    return result

def optimize_symmetric_ring(counts, bounds, seed=42, maxiter=50, popsize=10):
    """
    Optimize radii and offsets for a fixed set of ring counts.
    Returns best points and score.
    """
    n_params = len(counts) * 2 # radii and offsets
    # Flatten bounds
    flat_bounds = []
    for i in range(len(counts)):
        flat_bounds.append(bounds[i][0]) # radius
        flat_bounds.append((0, 2*np.pi)) # offset
    
    def neg_score_params(params):
        radii = params[0::2]
        offsets = params[1::2]
        points = generate_ring_points(counts, radii, offsets)
        return -compute_score(points)
    
    try:
        res = differential_evolution(
            neg_score_params,
            flat_bounds,
            seed=seed,
            maxiter=maxiter,
            tol=1e-6,
            popsize=popsize,
            polish=True
        )
        best_params = res.x
        radii = best_params[0::2]
        offsets = best_params[1::2]
        points = generate_ring_points(counts, radii, offsets)
        score = compute_score(points)
        return points, score
    except Exception:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = 142.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s
    
    # Phase 1: Generic Ring DE (kept from original, but maybe reduced budget)
    # We will prioritize specific symmetric structures which are known to be good.
    
    # Phase 1b: Targeted Symmetric Searches
    # Try 7+7
    if time.time() - start_time < deadline - 50:
        counts = [7, 7]
        bounds = [(0.5, 1.5), (0.1, 0.9)] # Radii bounds
        p, s = optimize_symmetric_ring(counts, bounds, seed=1, maxiter=100, popsize=15)
        if p is not None:
            try_update(p)
            
    # Try 5+5+4
    if time.time() - start_time < deadline - 50:
        counts = [5, 5, 4]
        bounds = [(0.5, 1.5), (0.3, 1.0), (0.0, 0.6)]
        p, s = optimize_symmetric_ring(counts, bounds, seed=2, maxiter=100, popsize=15)
        if p is not None:
            try_update(p)
            
    # Try 4+4+4+2
    if time.time() - start_time < deadline - 50:
        counts = [4, 4, 4, 2]
        bounds = [(0.5, 1.5), (0.4, 1.1), (0.2, 0.7), (0.0, 0.3)]
        p, s = optimize_symmetric_ring(counts, bounds, seed=3, maxiter=100, popsize=15)
        if p is not None:
            try_update(p)
            
    # Try 6+4+4
    if time.time() - start_time < deadline - 50:
        counts = [6, 4, 4]
        bounds = [(0.5, 1.5), (0.3, 1.0), (0.1, 0.6)]
        p, s = optimize_symmetric_ring(counts, bounds, seed=4, maxiter=100, popsize=15)
        if p is not None:
            try_update(p)
            
    # Try 8+6
    if time.time() - start_time < deadline - 50:
        counts = [8, 6]
        bounds = [(0.5, 1.5), (0.2, 0.8)]
        p, s = optimize_symmetric_ring(counts, bounds, seed=5, maxiter=100, popsize=15)
        if p is not None:
            try_update(p)

    # Phase 2: Multi-start L-BFGS-B on full 28D space from good candidates
    if best_points is not None and time.time() - start_time < deadline - 40:
        # Generate diverse starting points
        starts = [best_points]
        
        # Add jittered versions
        for _ in range(8): # Increased jitter starts
            jittered = best_points + rng.normal(0, 0.1, (N, 2))
            starts.append(jittered)
        
        # Add some structured starts (in case DE missed them)
        # Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        # Two rings 7+7 (optimized angles)
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.55*np.cos(a_in), 0.55*np.sin(a_in)))
        )))
        
        # Three rings 5+5+4
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = a1 + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.65*np.cos(a2), 0.65*np.sin(a2))),
            np.column_stack((0.3*np.cos(a3), 0.3*np.sin(a3)))
        )))
        
        for start in starts:
            if time.time() - start_time > deadline - 20:
                break
            
            x0 = start.flatten()
            
            # Annealing schedule for tau
            taus = [0.05, 0.02, 0.008, 0.003, 0.001]
            
            current_x = x0
            for tau in taus:
                if time.time() - start_time > deadline - 12:
                    break
                
                try:
                    res = minimize(
                        soft_min_score,
                        current_x,
                        args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass
    
    # Phase 3: Basin hopping from best
    if best_points is not None and time.time() - start_time < deadline - 18:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10},
                minsteps=5, T=0.5, stepsize=0.25,
                interval=5, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass
    
    # Phase 4: Powell refinement
    if best_points is not None and time.time() - start_time < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish
    if best_points is not None and time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    return best_points
