# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

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
    bary: (N, 2) array of (u, v). w = 1 - u - v.
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

def _param_to_bary(params):
    """
    Map unconstrained parameters (2N) to barycentric coordinates in the interior.
    Uses a softplus-like mapping to ensure u>0, v>0, u+v<1.
    Let z1, z2 be parameters for point i.
    u = sigmoid(z1) * 0.9
    v = sigmoid(z2) * 0.9 * (1 - sigmoid(z1))
    This ensures u < 0.9, v < 0.9, and u+v < 0.9 + 0.9*0.1 < 1.
    Actually, simpler:
    Let a = sigmoid(z1), b = sigmoid(z2).
    u = a * (1-b)
    v = b * (1-a)
    w = 1 - u - v = 1 - a(1-b) - b(1-a) = 1 - a + ab - b + ab = 1 - a - b + 2ab.
    This doesn't guarantee positivity of w easily.
    
    Better: Use softmax on 3 coordinates.
    Let z = [z1, z2, z3]. But we only have 2 params per point.
    We can fix one parameter or use a specific mapping.
    
    Let's use:
    u = exp(p1) / (exp(p1) + exp(p2) + 1)
    v = exp(p2) / (exp(p1) + exp(p2) + 1)
    w = 1 / (exp(p1) + exp(p2) + 1)
    This guarantees u,v,w > 0 and u+v+w=1.
    """
    p1 = params[0::2]
    p2 = params[1::2]
    
    # Clip to prevent overflow in exp
    p1 = np.clip(p1, -100, 100)
    p2 = np.clip(p2, -100, 100)
    
    e1 = np.exp(p1)
    e2 = np.exp(p2)
    denom = e1 + e2 + 1.0
    
    u = e1 / denom
    v = e2 / denom
    
    return np.column_stack((u, v))

def _bary_to_param(bary):
    """
    Inverse of _param_to_bary.
    u = e1 / (e1+e2+1) => u(e1+e2+1) = e1 => e1(1-u) = u(e2+1) => e1 = u(e2+1)/(1-u)
    v = e2 / (e1+e2+1) => v(e1+e2+1) = e2 => e2(1-v) = v(e1+1) => e2 = v(e1+1)/(1-v)
    This is a system of two equations.
    Alternatively, we can just use the log of the ratios.
    e1/e2 = (u/v) * (e2+1)/(e1+1) ... complicated.
    
    Simpler: Since we only need initial params for optimization, we can approximate.
    p1 = log(u) - log(1-u-v)
    p2 = log(v) - log(1-u-v)
    Let's check if _param_to_bary(_bary_to_param(bary)) ~ bary.
    Let w = 1-u-v.
    p1 = log(u/w), p2 = log(v/w).
    e1 = u/w, e2 = v/w.
    denom = u/w + v/w + 1 = (u+v+w)/w = 1/w.
    u_calc = (u/w) / (1/w) = u.
    v_calc = (v/w) / (1/w) = v.
    Yes, this works perfectly.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Clip to avoid log(0)
    u = np.clip(u, 1e-12, 1.0)
    v = np.clip(v, 1e-12, 1.0)
    w = np.clip(w, 1e-12, 1.0)
    
    p1 = np.log(u / w)
    p2 = np.log(v / w)
    
    params = np.zeros(2 * N)
    params[0::2] = p1
    params[1::2] = p2
    
    return params

def _objective_riemannian(params, beta, lambda_pen=10.0):
    """
    Objective for Riemannian-like optimization.
    Minimizes -t + penalty for area violations.
    t is not a variable here, we just maximize the soft-min of areas.
    Soft-min of areas A_i is approx by:
    m - (1/beta) * log(sum(exp(-beta * (A_i - m))))
    We want to maximize this. So we minimize the negative.
    """
    bary = _param_to_bary(params)
    
    # Check validity (should always be valid by construction, but just in case)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    if np.any(u < 0) or np.any(v < 0) or np.any(w < 0):
        return 1e10
        
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    m = np.min(areas)
    shifted = areas - m
    # Soft min: m - (1/beta) * log(sum(exp(-beta * shifted)))
    # To avoid overflow, subtract max of exponents
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min

def _local_optimize_riemannian(x0, beta_schedule, max_iter_per_beta=100):
    """
    Run L-BFGS-B on the parameter space with annealing of beta.
    """
    current_params = x0.copy()
    
    for beta in beta_schedule:
        try:
            res = minimize(
                _objective_riemannian,
                current_params,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_params = res.x
        except Exception:
            pass
            
    return current_params

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 25
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
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()
        
    best_bary = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(_project_to_bary_simplex(pts))
        
    # 3. Magic Init (Grid-based)
    magic_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.25],
    ])
    inits.append(_project_to_bary_simplex(magic_pts))
    
    # 4. Another structured init
    bary_struct = np.array([
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
    inits.append(_project_to_bary_simplex(bary_struct))
    
    # Phase 1: Riemannian Optimization from structured inits
    beta_schedule = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for init_bary in inits:
        if time_left() < 5:
            break
            
        try:
            # Convert to parameters
            init_params = _bary_to_param(init_bary)
            
            # Run Riemannian optimization
            opt_params = _local_optimize_riemannian(init_params, beta_schedule, max_iter_per_beta=200)
            
            # Convert back to barycentric
            opt_bary = _param_to_bary(opt_params)
            
            # Evaluate
            score = _min_area_ratio_from_bary(opt_bary)
            
            if score > best_score:
                best_score = score
                best_bary = opt_bary
                
        except Exception:
            pass

    # Phase 2: Refinement with standard SLSQP/NM on the best result found
    if best_bary is not None and time_left() > 20:
        try:
            bounds = [(0, 1)] * (2 * N)
            
            def _objective_slsqp(x):
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

            # SLSQP polish
            res = minimize(
                _objective_slsqp, best_bary.flatten(), method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            cand_bary = _project_to_bary_simplex(res.x.reshape(N, 2))
            cand_score = _min_area_ratio_from_bary(cand_bary)
            
            if cand_score > best_score:
                best_score = cand_score
                best_bary = cand_bary

            # Nelder-Mead polish
            res_nm = minimize(
                _objective_slsqp, best_bary.flatten(), method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
            )
            cand_bary_nm = _project_to_bary_simplex(res_nm.x.reshape(N, 2))
            cand_score_nm = _min_area_ratio_from_bary(cand_bary_nm)
            
            if cand_score_nm > best_score:
                best_score = cand_score_nm
                best_bary = cand_bary_nm
                
        except Exception:
            pass

    # Phase 3: Random restarts with Riemannian optimization if time remains
    if time_left() > 30:
        rng2 = np.random.RandomState(999)
        for _ in range(10):
            if time_left() < 10:
                break
            try:
                pts = np.zeros((N, 2))
                for i in range(N):
                    u = rng2.random()
                    v = rng2.random()
                    while u + v > 1.0:
                        u = rng2.random()
                        v = rng2.random()
                    pts[i, 0] = u
                    pts[i, 1] = v
                init_bary = _project_to_bary_simplex(pts)
                
                init_params = _bary_to_param(init_bary)
                opt_params = _local_optimize_riemannian(init_params, beta_schedule[:3], max_iter_per_beta=100)
                opt_bary = _param_to_bary(opt_params)
                
                score = _min_area_ratio_from_bary(opt_bary)
                
                if score > best_score:
                    best_score = score
                    best_bary = opt_bary
            except Exception:
                pass

    if best_bary is None:
        best_bary = _structured_init_fps()
    
    cart_points = _bary_to_cart(best_bary[:, 0], best_bary[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
