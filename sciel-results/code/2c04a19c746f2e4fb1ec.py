import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _project_to_bary_simplex(bary):
    """Project onto the barycentric simplex: u>=0, v>=0, u+v<=1."""
    bary = np.array(bary, dtype=float)
    bary = np.maximum(bary, 0.0)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _all_dets_from_bary(bary):
    """Compute all triple determinants (2*area) from barycentric coords."""
    idx = _IDX_TRIPLES
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    du_ji = uj - ui
    dv_ji = vj - vi
    du_ki = uk - ui
    dv_ki = vk - vi
    
    dets = np.sqrt(3) / 2.0 * (du_ji * dv_ki - dv_ji * du_ki)
    return dets

def _min_area_ratio_from_bary(bary):
    """Compute the minimum area (divided by triangle area) from barycentric coords."""
    dets = _all_dets_from_bary(bary)
    min_det = np.min(np.abs(dets))
    return 0.5 * min_det / TRI_AREA

def _objective_smooth(x, beta=50.0):
    """Smooth objective for global optimizers: negative log-sum-exp of negative areas."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0
    
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    lse = max_nb + np.log(np.sum(np.exp(neg_beta_dets - max_nb)))
    return lse / beta

def _objective_exact(x):
    """Exact objective for local polishing: negative min area."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    return -np.min(np.abs(dets))

def _structured_init_fps():
    """Farthest-point-sampling structured initialization in barycentric coords."""
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    points.append(np.array([1.0/3, 1.0/3]))
    
    rng = np.random.RandomState(789)
    while len(points) < N:
        candidates = []
        for _ in range(2000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v <= 1.0:
                candidates.append(np.array([u, v]))
        candidates = np.array(candidates)
        
        best_candidate = None
        best_min_dist = -1
        for cand in candidates:
            dists = [np.linalg.norm(cand - p) for p in points]
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = cand
        if best_candidate is not None:
            points.append(best_candidate)
    return np.array(points)

def _simulated_annealing(x0, start_time, deadline):
    """Simulated annealing to refine the solution."""
    current_x = x0.copy()
    current_bary = _project_to_bary_simplex(current_x.reshape(-1, 2))
    current_score = _min_area_ratio_from_bary(current_bary)
    best_x = current_x.copy()
    best_score = current_score
    best_bary = current_bary.copy()
    
    T = 0.05
    cooling = 0.995
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5.0:
        # Perturb
        step_size = 0.05 * T
        x_pert = current_x + rng.normal(0, step_size, size=current_x.shape)
        bary_pert = _project_to_bary_simplex(x_pert.reshape(-1, 2))
        score_pert = _min_area_ratio_from_bary(bary_pert)
        
        delta = score_pert - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_x = x_pert
            current_bary = bary_pert
            current_score = score_pert
            
            if current_score > best_score:
                best_score = current_score
                best_x = current_x.copy()
                best_bary = current_bary.copy()
        
        T *= cooling
        if T < 1e-6:
            break
            
    return best_x, best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Initial structured start
    bary_init = _structured_init_fps()
    x_init = bary_init.flatten()
    
    best_x = x_init.copy()
    best_bary = bary_init.copy()
    best_score = _min_area_ratio_from_bary(best_bary)
    
    # Phase 1: Differential Evolution (Global Search)
    # Bounds: u in [0,1], v in [0,1]. The projection handles u+v<=1.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        if time_left() > 60.0:
            de_result = differential_evolution(
                _objective_smooth,
                bounds=bounds,
                args=(30.0,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                polish=False,  # We do our own polish
                init='sobol'
            )
            if de_result.success or de_result.fun < _objective_smooth(best_x, 30.0):
                x_de = de_result.x
                bary_de = _project_to_bary_simplex(x_de.reshape(-1, 2))
                score_de = _min_area_ratio_from_bary(bary_de)
                if score_de > best_score:
                    best_score = score_de
                    best_x = x_de.copy()
                    best_bary = bary_de.copy()
    except Exception:
        pass
    
    # Phase 2: Simulated Annealing (Refinement)
    if time_left() > 20.0:
        x_sa, bary_sa, score_sa = _simulated_annealing(best_x, start_time, deadline)
        if score_sa > best_score:
            best_score = score_sa
            best_x = x_sa.copy()
            best_bary = bary_sa.copy()
    
    # Phase 3: Local Polish with Nelder-Mead (Exact Objective)
    if time_left() > 5.0:
        try:
            nm_result = minimize(
                _objective_exact,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            x_nm = nm_result.x
            bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
            score_nm = _min_area_ratio_from_bary(bary_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_x = x_nm.copy()
                best_bary = bary_nm.copy()
        except Exception:
            pass
            
    # Final fallback
    if best_x is None:
        best_x = np.zeros(2 * N)
        
    return _bary_to_cart(_project_to_bary_simplex(best_x.reshape(-1, 2)))
