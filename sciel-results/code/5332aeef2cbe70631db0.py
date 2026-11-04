# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
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
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
    
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _all_areas_from_bary(bary):
    dets = _all_dets_from_bary(bary)
    return np.abs(dets)

def _objective_smooth(x, beta, penalty_lambda=100.0):
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
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

def _structured_init_fps():
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

def _project_to_bary_simplex(bary):
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _simulated_annealing(bary_init, n_iter=2000, T_start=0.05, T_end=0.001, rng=None):
    """
    Simulated annealing to escape local optima.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    current = bary_init.copy()
    current_score = _min_area_ratio_from_bary(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        # Random perturbation
        perturbation = rng.normal(0, T, size=(N, 2))
        candidate = current + perturbation
        candidate = _project_to_bary_simplex(candidate)
        
        candidate_score = _min_area_ratio_from_bary(candidate)
        
        if candidate_score >= current_score:
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        else:
            # Accept with probability exp((candidate_score - current_score) / T)
            delta = candidate_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current = candidate
                    current_score = candidate_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(25):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # Bounds for u, v
    bounds = [(0, 1)] * (2 * N)
    
    for init in inits:
        if time.time() > deadline - 15:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            # Phase 1: Simulated annealing to explore
            sa_time_limit = min(20.0, deadline - time.time() - 25)
            if sa_time_limit > 5:
                sa_rng = np.random.RandomState(int(time.time() * 1000) % 100000)
                sa_bary, sa_score = _simulated_annealing(init_bary, n_iter=1500, T_start=0.03, T_end=0.001, rng=sa_rng)
                if sa_score > best_score:
                    best_score = sa_score
                    best_x = sa_bary
            
            # Phase 2: Gradient-based refinement
            if time.time() > deadline - 15:
                break
                
            start_bary = best_x if (best_x is not None and best_score > 0) else init_bary
            
            best_local_x = start_bary
            best_local_score = _min_area_ratio_from_bary(start_bary)
            
            betas = [50.0, 200.0, 1000.0]
            current_x = start_bary.flatten()
            
            for beta in betas:
                if time.time() > deadline - 10:
                    break
                
                res = minimize(
                    _objective_smooth,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                x_opt = res.x.reshape(N, 2)
                x_opt = _project_to_bary_simplex(x_opt)
                
                score = _min_area_ratio_from_bary(x_opt)
                if score > best_local_score:
                    best_local_score = score
                    best_local_x = x_opt
                    current_x = x_opt.flatten()
            
            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
                
        except Exception:
            pass

    if best_x is None:
        best_x = _structured_init_fps()
    
    # Final high-precision polish pass
    if time.time() < deadline - 8:
        try:
            polish_x = best_x.flatten()
            for beta in [5000.0, 20000.0]:
                if time.time() > deadline - 4:
                    break
                res = minimize(
                    _objective_smooth,
                    polish_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                polish_x = res.x
                polish_bary = _project_to_bary_simplex(polish_x.reshape(N, 2))
                score = _min_area_ratio_from_bary(polish_bary)
                if score > best_score:
                    best_score = score
                    best_x = polish_bary
        except Exception:
            pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
