# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
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
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B using log-sum-exp soft-min.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess_u = np.maximum(0, -u)
    excess_v = np.maximum(0, -v)
    excess_w = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * (np.sum(excess_u**2) + np.sum(excess_v**2) + np.sum(excess_w**2))
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # Soft min: m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_exact(x):
    """
    Objective for SLSQP/NM: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 30
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    # Start with centroid
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

def _local_optimize_nm(x0, bounds, maxiter=200, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_exact, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, bounds, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_exact, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, bounds, maxiter=200, beta=500.0):
    """Run L-BFGS-B local optimization from x0 using smooth objective."""
    try:
        res = minimize(
            _objective_smooth, x0, args=(beta,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(20):
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
        
    # 3. Heuristic Grid Init (Symmetric)
    # 3 corners, 3 edge midpoints, 3 inner ring, 1 center? 
    # Let's try a structured grid
    bary_grid = []
    # Corners
    bary_grid.append([0.0, 0.0])
    bary_grid.append([1.0, 0.0])
    bary_grid.append([0.0, 1.0])
    # Edge midpoints
    bary_grid.append([0.5, 0.0])
    bary_grid.append([0.0, 0.5])
    bary_grid.append([0.5, 0.5])
    # Inner points
    bary_grid.append([1/3, 1/3])
    bary_grid.append([0.25, 0.25])
    bary_grid.append([0.5, 0.25])
    bary_grid.append([0.25, 0.5])
    bary_grid.append([0.75, 0.25]) # Wait, 0.75+0.25=1, on edge
    # Let's use a slightly perturbed grid
    bary_grid = np.array(bary_grid[:N])
    bary_grid = _project_to_bary_simplex(bary_grid)
    inits.append(bary_grid.flatten())
    
    # 4. Another structured start
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    bounds = [(0, 1)] * (2 * N)
    
    for init in inits:
        if time_left() < 5:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            best_local_x = init_bary
            best_local_score = _min_area_ratio_from_bary(init_bary)
            
            # Strategy: Multi-method refinement loop
            # 1. L-BFGS-B with Smooth Objective (Good for escaping local minima if beta is low)
            if time_left() > 10:
                current_x = init_bary.flatten()
                # Annealing beta
                for beta in [50.0, 200.0, 500.0]:
                    if time_left() < 5:
                        break
                    x_opt = _local_optimize_lbfgs(current_x, bounds, maxiter=100, beta=beta)
                    bary_opt = _project_to_bary_simplex(x_opt.reshape(N, 2))
                    score = _min_area_ratio_from_bary(bary_opt)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_x = bary_opt
                        current_x = bary_opt.flatten()

            # 2. SLSQP (Precise, handles non-smoothness better with constraints)
            if time_left() > 10:
                x_slsqp = _local_optimize_slsqp(best_local_x.flatten(), bounds, maxiter=200)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                if score_slsqp > best_local_score:
                    best_local_score = score_slsqp
                    best_local_x = bary_slsqp

            # 3. Nelder-Mead (Global-ish local search)
            if time_left() > 10:
                x_nm = _local_optimize_nm(best_local_x.flatten(), bounds, maxiter=300)
                bary_nm = _project_to_bary_simplex(x_nm.reshape(N, 2))
                score_nm = _min_area_ratio_from_bary(bary_nm)
                if score_nm > best_local_score:
                    best_local_score = score_nm
                    best_local_x = bary_nm
                    
            # 4. Final Polish with SLSQP
            if time_left() > 5:
                x_final = _local_optimize_slsqp(best_local_x.flatten(), bounds, maxiter=100)
                bary_final = _project_to_bary_simplex(x_final.reshape(N, 2))
                score_final = _min_area_ratio_from_bary(bary_final)
                if score_final > best_local_score:
                    best_local_score = score_final
                    best_local_x = bary_final

            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
                
        except Exception:
            pass

    # If we have time, do a few random restarts with SLSQP
    if time_left() > 20:
        for _ in range(15):
            if time_left() < 10:
                break
            try:
                pts = np.zeros((N, 2))
                for i in range(N):
                    u = rng.random()
                    v = rng.random()
                    while u + v > 1.0:
                        u = rng.random()
                        v = rng.random()
                    pts[i, 0] = u
                    pts[i, 1] = v
                init_bary = _project_to_bary_simplex(pts)
                
                # Quick LBFGS then SLSQP
                x_lbfgs = _local_optimize_lbfgs(init_bary.flatten(), bounds, maxiter=50, beta=100.0)
                bary_lbfgs = _project_to_bary_simplex(x_lbfgs.reshape(N, 2))
                
                x_slsqp = _local_optimize_slsqp(bary_lbfgs.flatten(), bounds, maxiter=100)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                
                if score_slsqp > best_score:
                    best_score = score_slsqp
                    best_x = bary_slsqp
            except Exception:
                pass

    if best_x is None:
        best_x = _structured_init_fps()
        
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
