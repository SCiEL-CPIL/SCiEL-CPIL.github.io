"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    """Exact score: min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_tri_areas(pts)) / ha)

def _soft_min(areas, tau):
    """
    Compute smooth minimum using log-sum-exp.
    """
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _neg_soft_min_ratio(pts, temp):
    """
    Smooth surrogate to maximize: -soft_min(areas) / hull_area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    a = _tri_areas(pts)
    a = np.maximum(a, 1e-15)
    s = -a / temp
    s = s - np.max(s)
    logsum = np.log(np.sum(np.exp(s)))
    soft_min = -logsum * temp
    return soft_min / ha

def _decode_params(x):
    """
    Decode parameter vector x into 2D points.
    x[0:N] are log-radii (to ensure positivity), x[N:2*N] are angles.
    Sorts points by angle to enforce convex order.
    """
    log_radii = x[:N]
    angles = x[N:2*N]
    
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    # Sort by angle
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_log_radii = log_radii[sort_idx]
    
    # Exponentiate to get positive radii
    radii = np.exp(sorted_log_radii)
    
    pts = np.column_stack([
        radii * np.cos(sorted_angles),
        radii * np.sin(sorted_angles)
    ])
    return pts

def _objective(x, tau=0.05):
    """
    Objective function: Negative smooth minimum triangle area normalized by hull area.
    Uses log-sum-exp for smoothness.
    """
    pts = _decode_params(x)
    
    # Compute all triangle areas
    n = N
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas == 0):
        return 1e10
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10

    # Smooth min using log-sum-exp
    smooth_min = _soft_min(areas, tau)
    score = smooth_min / hull_area
    return -score

def _construct_initial_params(seed):
    """
    Generate initial parameters (log-radii, angles) for optimization.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a regular polygon-like structure
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
    log_radii = np.zeros(N) + rng.normal(0, 0.1, N)
    
    x = np.concatenate([log_radii, angles])
    return x

def _refine_cartesian(pts, time_budget):
    """
    Refine a Cartesian point set using L-BFGS on a smooth surrogate.
    This acts as the local polish step.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    # Use a moderately small temperature for the soft-min to guide L-BFGS
    # towards the exact min without being too noisy.
    temp = 0.005
    
    def obj(x):
        p = x.reshape(N, 2)
        return _neg_soft_min_ratio(p, temp)
    
    try:
        res = minimize(obj, best.flatten(), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand.copy()
    except Exception:
        pass
        
    # Second pass with tighter temperature if time permits
    if time.time() - t0 < time_budget - 1:
        temp2 = 0.001
        def obj2(x):
            p = x.reshape(N, 2)
            return _neg_soft_min_ratio(p, temp2)
        try:
            res = minimize(obj2, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
            
    return best, best_s

def _differential_evolution_angular(r0, a0, time_budget, rng):
    """
    Global search in angular (radius, angle) space using Differential Evolution.
    Parameters:
    r0: initial radii (N,)
    a0: initial angles (N,)
    time_budget: seconds to spend
    rng: random state
    """
    t0 = time.time()
    deadline = t0 + time_budget
    
    # Bounds for radii and angles
    r_min, r_max = 0.0, 1.5
    a_min, a_max = 0.0, 2 * np.pi
    
    # Population size
    pop_size = 20
    
    # Initialize population
    pop_r = np.zeros((pop_size, N))
    pop_a = np.zeros((pop_size, N))
    
    pop_r[0] = r0
    pop_a[0] = a0
    
    for i in range(1, pop_size):
        pop_r[i] = rng.uniform(0.1, 1.2, N)
        pop_a[i] = rng.uniform(0, 2 * np.pi, N)
        
    # Evaluate initial population
    pop_scores = np.zeros(pop_size)
    for i in range(pop_size):
        pop_scores[i] = _evaluate_angular(pop_r[i], pop_a[i])
        
    best_idx = np.argmax(pop_scores)
    best_r = pop_r[best_idx].copy()
    best_a = pop_a[best_idx].copy()
    best_score = pop_scores[best_idx]
    
    # DE Parameters
    F = 0.8
    CR = 0.9
    
    # Main DE Loop
    while time.time() < deadline - 5:
        new_pop_r = np.zeros_like(pop_r)
        new_pop_a = np.zeros_like(pop_a)
        new_scores = np.zeros(pop_size)
        
        for i in range(pop_size):
            idxs = rng.choice(pop_size, 3, replace=False)
            a_idx, b_idx, c_idx = idxs
            
            v_r = pop_r[a_idx] + F * (pop_r[b_idx] - pop_r[c_idx])
            v_a = pop_a[a_idx] + F * (pop_a[b_idx] - pop_a[c_idx])
            
            v_a = v_a % (2 * np.pi)
            
            j_rand = rng.randint(0, N)
            trial_r = np.zeros(N)
            trial_a = np.zeros(N)
            
            for j in range(N):
                if rng.random() < CR or j == j_rand:
                    trial_r[j] = v_r[j]
                    trial_a[j] = v_a[j]
                else:
                    trial_r[j] = pop_r[i][j]
                    trial_a[j] = pop_a[i][j]
                    
            trial_r = np.clip(trial_r, r_min, r_max)
            
            trial_score = _evaluate_angular(trial_r, trial_a)
            
            if trial_score >= pop_scores[i]:
                new_pop_r[i] = trial_r
                new_pop_a[i] = trial_a
                new_scores[i] = trial_score
            else:
                new_pop_r[i] = pop_r[i]
                new_pop_a[i] = pop_a[i]
                new_scores[i] = pop_scores[i]
                
        pop_r = new_pop_r
        pop_a = new_pop_a
        pop_scores = new_scores
        
        current_best_idx = np.argmax(pop_scores)
        if pop_scores[current_best_idx] > best_score:
            best_score = pop_scores[current_best_idx]
            best_r = pop_r[current_best_idx].copy()
            best_a = pop_a[current_best_idx].copy()
            
    best_pts = _angular_to_cartesian(best_r, best_a)
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        best_pts, best_score = _refine_cartesian(best_pts, remaining_time)
        
    return best_pts, best_score

def _evaluate_angular(radii, angles):
    """Evaluate score given polar parameters."""
    pts = _angular_to_cartesian(radii, angles)
    return _score(pts)

def _angular_to_cartesian(radii, angles):
    """Convert polar coordinates to Cartesian."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def construct():
    """Main entry: Multi-start SLSQP in Angular Space with Local Polish."""
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin for 150s limit
    
    # Initial points: Regular polygon
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_init = np.ones(N)
    init_pts = np.column_stack([r_init * np.cos(th), r_init * np.sin(th)])
    
    best_pts = init_pts.copy()
    best_score = _score(init_pts)
    
    # Multi-start with SLSQP using log-barrier formulation
    num_starts = 15
    time_per_start = (deadline - t0) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        x0 = _construct_initial_params(seed=1000 + i)
        
        try:
            bounds = [(-5.0, 5.0)] * N + [(0.0, 2 * np.pi)] * N
            
            res = minimize(
                _objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success or res.fun < -best_score:
                x_opt = res.x
                pts_opt = _decode_params(x_opt)
                score_opt = _score(pts_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
                    
        except Exception:
            pass
    
    # If SLSQP didn't find a good solution, try L-BFGS-B as fallback
    if best_score < 0.01:
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
                
            x0 = _construct_initial_params(seed=2000 + i)
            
            try:
                bounds = [(-5.0, 5.0)] * N + [(0.0, 2 * np.pi)] * N
                
                res = minimize(
                    _objective, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < -best_score:
                    x_opt = res.x
                    pts_opt = _decode_params(x_opt)
                    score_opt = _score(pts_opt)
                    
                    if score_opt > best_score:
                        best_score = score_opt
                        best_pts = pts_opt
                        
            except Exception:
                pass

    # Final polish on the absolute best found
    remaining = deadline - time.time() - 2
    if remaining > 5:
        try:
            pts, s = _refine_cartesian(best_pts, remaining)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
            
    return best_pts
