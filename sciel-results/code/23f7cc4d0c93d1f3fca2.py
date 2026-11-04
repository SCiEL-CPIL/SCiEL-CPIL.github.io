# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools
import random

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
NUM_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    # P = u*(0,0) + v*(1,0) + w*(0.5, sqrt(3)/2)
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex u>=0, v>=0, u+v<=1.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _all_dets_from_bary(bary):
    """
    Compute all determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    bary: (N, 2) array of (u, v). w is derived.
    Returns the minimum area ratio of any triangle formed by 3 points.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
    
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _all_areas_from_bary(bary):
    """
    Compute all 165 triangle areas.
    """
    dets = _all_dets_from_bary(bary)
    return np.abs(dets)

def _objective_raw(x):
    """
    Raw objective for SLSQP/NM: minimize -min_area.
    """
    bary = x.reshape(N, 2)
    # Project to ensure validity if optimizer strays slightly
    bary = _project_to_bary_simplex(bary)
    areas = _all_areas_from_bary(bary)
    if len(areas) == 0:
        return 0.0
    return -np.min(areas)

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    areas = _all_areas_from_bary(bary)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _local_optimize_slsqp(x_init, deadline):
    """
    Local optimization using SLSQP on the raw objective.
    """
    if time.time() > deadline - 5:
        return x_init, _min_area_ratio_from_bary(x_init)
    
    bounds = [(0, 1)] * (2 * N)
    
    try:
        res = minimize(
            _objective_raw,
            x_init.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        x_opt = res.x.reshape(N, 2)
        x_opt = _project_to_bary_simplex(x_opt)
        score = _min_area_ratio_from_bary(x_opt)
        return x_opt, score
    except Exception:
        return x_init, _min_area_ratio_from_bary(x_init)

def _local_optimize_nm(x_init, deadline):
    """
    Local optimization using Nelder-Mead.
    """
    if time.time() > deadline - 5:
        return x_init, _min_area_ratio_from_bary(x_init)
    
    try:
        res = minimize(
            _objective_raw,
            x_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
        )
        x_opt = res.x.reshape(N, 2)
        x_opt = _project_to_bary_simplex(x_opt)
        score = _min_area_ratio_from_bary(x_opt)
        return x_opt, score
    except Exception:
        return x_init, _min_area_ratio_from_bary(x_init)

def _local_optimize_lbfgs(x_init, deadline):
    """
    Local optimization using L-BFGS-B with smooth objective.
    """
    if time.time() > deadline - 5:
        return x_init, _min_area_ratio_from_bary(x_init)
    
    bounds = [(0, 1)] * (2 * N)
    
    best_x = x_init
    best_score = _min_area_ratio_from_bary(x_init)
    
    # Use increasing beta for sharper approximation
    for beta in [50.0, 200.0, 1000.0]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _objective_smooth,
                best_x.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            x_opt = res.x.reshape(N, 2)
            x_opt = _project_to_bary_simplex(x_opt)
            score = _min_area_ratio_from_bary(x_opt)
            if score > best_score:
                best_score = score
                best_x = x_opt
        except Exception:
            continue
            
    return best_x, best_score

def _simulated_annealing(x_init, deadline, rng):
    """
    Simulated Annealing search.
    """
    current_x = x_init.copy()
    current_score = _min_area_ratio_from_bary(current_x)
    best_x = current_x.copy()
    best_score = current_score
    
    T = 0.05
    cooling_rate = 0.95
    step_size = 0.05
    
    while time.time() < deadline - 20:
        # Generate neighbor
        x_new = current_x.copy()
        # Perturb a random subset of points
        num_perturb = rng.randint(1, 3)
        idxs = rng.choice(N, num_perturb, replace=False)
        noise = rng.normal(0, step_size, (num_perturb, 2))
        x_new[idxs] += noise
        x_new = _project_to_bary_simplex(x_new)
        
        new_score = _min_area_ratio_from_bary(x_new)
        
        # Acceptance criterion
        if new_score > current_score:
            current_x = x_new
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_x = x_new.copy()
        else:
            delta = current_score - new_score
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_x = x_new
                current_score = new_score
        
        # Cooling
        T *= cooling_rate
        
        # Periodic local refinement to escape shallow local minima
        if rng.random() < 0.05: # 5% chance
            # Choose a random local optimizer
            opt_method = rng.choice(['slsqp', 'nm', 'lbfgs'])
            if opt_method == 'slsqp':
                refined_x, refined_score = _local_optimize_slsqp(current_x, deadline)
            elif opt_method == 'nm':
                refined_x, refined_score = _local_optimize_nm(current_x, deadline)
            else:
                refined_x, refined_score = _local_optimize_lbfgs(current_x, deadline)
                
            if refined_score > best_score:
                best_score = refined_score
                best_x = refined_x.copy()
            # Update current to the refined position if it's better than current
            if refined_score > current_score:
                current_x = refined_x
                current_score = refined_score
            else:
                # If refinement didn't improve current, keep current but maybe reset temp slightly?
                # Or just continue. Let's keep current.
                pass
                
    return best_x, best_score

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - selected, axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def construct():
    start_time = time.time()
    # Safety margin: return 10s before 120s limit
    deadline = start_time + 110.0
    
    best_x = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # 1. Generate diverse initial configurations
    inits = []
    
    # FPS Init
    inits.append(_structured_init_fps())
    
    # Random Inits
    for _ in range(5):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts)
        
    # Grid-like Init
    grid_pts = []
    for i in range(3):
        for j in range(3):
            if len(grid_pts) < N:
                u = (i + 0.5) / 3.0
                v = (j + 0.5) / 3.0
                if u + v <= 1.0:
                    grid_pts.append([u, v])
    # Fill remaining randomly if grid is small
    while len(grid_pts) < N:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            grid_pts.append([u, v])
    inits.append(np.array(grid_pts[:N]))
    
    # Process each init
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        try:
            init_bary = _project_to_bary_simplex(init)
            
            # Phase 1: Local refinement to get a good starting point for SA
            # Use L-BFGS-B for speed
            x_refined, score_refined = _local_optimize_lbfgs(init_bary, deadline - 20)
            
            if score_refined > best_score:
                best_score = score_refined
                best_x = x_refined.copy()
            
            # Phase 2: Simulated Annealing starting from refined point
            if time.time() < deadline - 20:
                x_sa, score_sa = _simulated_annealing(x_refined, deadline - 10, rng)
                
                if score_sa > best_score:
                    best_score = score_sa
                    best_x = x_sa.copy()
                    
        except Exception:
            continue
    
    # Final Polish: Apply all local optimizers to the best solution found
    if best_x is not None and time.time() < deadline - 5:
        current_best = best_x.copy()
        current_best_score = best_score
        
        # Try SLSQP
        x_sl, s_sl = _local_optimize_slsqp(current_best, deadline - 5)
        if s_sl > current_best_score:
            current_best = x_sl
            current_best_score = s_sl
            
        # Try NM
        x_nm, s_nm = _local_optimize_nm(current_best, deadline - 5)
        if s_nm > current_best_score:
            current_best = x_nm
            current_best_score = s_nm
            
        # Try L-BFGS
        x_lb, s_lb = _local_optimize_lbfgs(current_best, deadline)
        if s_lb > current_best_score:
            current_best = x_lb
            current_best_score = s_lb
            
        best_x = current_best
        best_score = current_best_score
    
    if best_x is None:
        # Fallback
        best_x = _structured_init_fps()
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
