"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Uses radial perturbation of regular 13-gon (13 DOF) with L-BFGS-B + simulated annealing
+ coordinate descent for final polish."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        rest = pts[i+1:]
        m = len(rest)
        v1 = rest[:, np.newaxis, :] - pi
        v2 = rest[np.newaxis, :, :] - pi
        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
        idx = np.triu_indices(m, k=1)
        areas.append(np.abs(cross[idx]))
    areas = np.concatenate(areas)
    return 0.5 * areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

def _radial_to_points(radii, base_angle_offset=0.0):
    """Convert radial offsets to 2D points on a regular 13-gon.
    radii: array of shape (N,) with values in [0.3, 1.5]
    Returns points of shape (N, 2)."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + base_angle_offset
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _neg_radial_score(radii, beta=100.0, base_angle_offset=0.0):
    """Negative soft-min score for radial parametrization (for minimization)."""
    try:
        pts = _radial_to_points(radii, base_angle_offset)
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        # Soft min using log-sum-exp
        max_a = np.max(areas)
        shifted = areas - max_a
        log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + (-beta * max_a)
        soft_min = -log_sum_exp / beta
        return -soft_min / hull_area
    except Exception:
        return 0.0

def _neg_exact_radial_score(radii, base_angle_offset=0.0):
    """Negative exact score for radial parametrization."""
    try:
        pts = _radial_to_points(radii, base_angle_offset)
        return -_score(pts)
    except Exception:
        return 0.0

def _optimize_radial(radii_init, time_budget=10.0, start_time=None):
    """Optimize radial parameters using L-BFGS-B with progressive beta."""
    if start_time is None:
        start_time = time.time()
    
    best_radii = radii_init.copy()
    best_score = _score(_radial_to_points(radii_init))
    
    betas = [20, 50, 100, 200, 400]
    
    for beta in betas:
        if time.time() > start_time + time_budget - 2:
            break
        
        remaining = start_time + time_budget - 2 - time.time()
        if remaining < 1:
            break
        
        stage_time = min(remaining * 0.3, 5.0)
        
        result = minimize(
            _neg_radial_score,
            best_radii,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(0.3, 1.5)] * N,
            options={
                'maxiter': 300,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 40
            }
        )
        
        if time.time() > start_time + time_budget - 1:
            break
        
        trial_pts = _radial_to_points(result.x)
        trial_score = _score(trial_pts)
        
        if trial_score > best_score:
            best_score = trial_score
            best_radii = result.x.copy()
    
    return best_radii, best_score

def _coordinate_descent_full(pts, time_budget=15.0, start_time=None):
    """Coordinate descent: optimize each point individually using Nelder-Mead."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for round_num in range(3):
        if time.time() > deadline - 2:
            break
        
        for pt_idx in range(N):
            if time.time() > deadline - 1:
                break
            
            fixed_pts = best_pts.copy()
            x0 = fixed_pts[pt_idx].copy()
            
            def obj_single(x, idx=pt_idx, fp=fixed_pts):
                trial = fp.copy()
                trial[idx] = x
                return -_score(trial)
            
            res = minimize(
                obj_single,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if time.time() > deadline - 0.5:
                break
            
            if -res.fun > best_score + 1e-12:
                best_pts[pt_idx] = res.x
                best_score = -res.fun
    
    return best_pts, best_score

def _simulated_annealing_radial(radii_init, time_budget=20.0, start_time=None):
    """Simulated annealing on radial parameters."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    best_radii = radii_init.copy()
    best_score = _score(_radial_to_points(best_radii))
    current_radii = radii_init.copy()
    current_score = best_score
    
    T_start = 0.08
    T_end = 0.005
    n_steps = 1500
    
    rng = np.random.RandomState(999)
    
    for step in range(n_steps):
        if time.time() > deadline - 1:
            break
        
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        trial_radii = current_radii.copy()
        n_perturb = rng.randint(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        for i in idx:
            trial_radii[i] += T * rng.randn()
            trial_radii[i] = np.clip(trial_radii[i], 0.3, 1.5)
        
        trial_score = _score(_radial_to_points(trial_radii))
        
        if trial_score > current_score:
            current_radii = trial_radii
            current_score = trial_score
            if trial_score > best_score:
                best_score = trial_score
                best_radii = trial_radii.copy()
        else:
            delta = current_score - trial_score
            if delta > 0:
                accept_prob = np.exp(-delta / (T + 1e-15))
                if rng.random() < accept_prob:
                    current_radii = trial_radii
                    current_score = trial_score
    
    return best_radii, best_score

def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_temp = i
        while i_temp > 0:
            f /= base
            x += f * (i_temp % base)
            i_temp //= base
        result.append(x)
    return np.array(result)

def _generate_radial_inits():
    """Generate diverse initial radial configurations."""
    inits = []
    
    # 1. Uniform radii
    inits.append(np.ones(N))
    
    # 2. Slight alternating pattern
    alt = np.ones(N)
    alt[::2] = 1.05
    alt[1::2] = 0.95
    inits.append(alt)
    
    # 3. Halton-based radii
    h = _halton_sequence(N, 2)
    inits.append(0.7 + 0.6 * h)
    
    # 4. Another Halton
    h2 = _halton_sequence(N, 3)
    inits.append(0.7 + 0.6 * h2)
    
    # 5. Random
    rng = np.random.RandomState(42)
    inits.append(0.7 + 0.6 * rng.random(N))
    
    # 6. Another random
    rng2 = np.random.RandomState(123)
    inits.append(0.7 + 0.6 * rng2.random(N))
    
    # 7. Gradient pattern
    grad = np.linspace(0.8, 1.2, N)
    inits.append(grad)
    
    # 8. Bimodal
    bi = np.ones(N)
    bi[:6] = 1.1
    bi[6:] = 0.9
    inits.append(bi)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Phase 1: Optimize radial parameters from multiple initializations
    inits = _generate_radial_inits()
    
    best_pts = None
    best_score = -1
    best_radii = None
    
    for i, radii_init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        
        remaining = deadline - 35 - time.time()
        if remaining < 5:
            break
        
        stage_budget = min(remaining * 0.15, 8.0)
        
        radii_opt, score_opt = _optimize_radial(radii_init, time_budget=stage_budget, start_time=start_time)
        
        if score_opt > best_score:
            best_score = score_opt
            best_pts = _radial_to_points(radii_opt)
            best_radii = radii_opt
    
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
        best_radii = np.ones(N)
    
    # Phase 2: Simulated annealing on radial params to escape local minima
    remaining = deadline - 25 - time.time()
    if remaining > 8 and best_radii is not None:
        sa_budget = min(remaining * 0.4, 15.0)
        radii_sa, score_sa = _simulated_annealing_radial(best_radii, time_budget=sa_budget, start_time=start_time)
        if score_sa > best_score:
            best_score = score_sa
            best_pts = _radial_to_points(radii_sa)
            best_radii = radii_sa
    
    # Phase 3: Coordinate descent on full 2D points for precision
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        cd_budget = min(remaining * 0.6, 12.0)
        pts_cd, score_cd = _coordinate_descent_full(best_pts, time_budget=cd_budget, start_time=start_time)
        if score_cd > best_score:
            best_score = score_cd
            best_pts = pts_cd
    
    # Phase 4: Final radial optimization from the polished points
    remaining = deadline - 5 - time.time()
    if remaining > 3:
        # Extract approximate radii from best_pts
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Find best angle offset
        best_offset = 0
        best_offset_score = -1
        for offset in np.linspace(0, 2*np.pi/N, 4, endpoint=False):
            trial_pts = _radial_to_points(np.ones(N), offset)
            # Project best_pts onto radial basis
            radii_approx = np.array([
                np.dot(best_pts[j], np.array([np.cos(angles[j]+offset), np.sin(angles[j]+offset)]))
                for j in range(N)
            ])
            radii_approx = np.clip(radii_approx, 0.3, 1.5)
            s = _score(_radial_to_points(radii_approx, offset))
            if s > best_offset_score:
                best_offset_score = s
                best_offset = offset
        
        if best_offset_score > 0:
            radii_final, score_final = _optimize_radial(
                np.clip(radii_approx, 0.3, 1.5),
                time_budget=min(remaining * 0.7, 4.0),
                start_time=start_time
            )
            if score_final > best_score:
                best_score = score_final
                best_pts = _radial_to_points(radii_final, best_offset)
    
    # Phase 5: Final coordinate descent polish
    remaining = deadline - 2 - time.time()
    if remaining > 2:
        pts_final, score_final = _coordinate_descent_full(best_pts, time_budget=min(remaining, 3.0), start_time=start_time)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    return best_pts

# EVOLVE-BLOCK-END
