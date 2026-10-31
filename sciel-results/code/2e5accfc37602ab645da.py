import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2
TIME_LIMIT = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    iu = np.triu_indices(N, 1)
    i_min, j_min = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[i_min]
    p1 = P[j_min]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return None
    
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (P - p0) @ R.T / norm_v
    
    perm = list(range(N))
    perm[0] = i_min
    perm[1] = j_min
    remaining = [i for i in range(N) if i != i_min and i != j_min]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    params = transformed[2:].flatten()
    return params

def _reconstruct(params):
    """Reconstruct points from params (p0=(0,0), p1=(1,0), rest=params)."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return P

def _exact_objective(params):
    """Exact objective in reduced parameter space."""
    P = _reconstruct(params)
    return _score(P)

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Crown configuration: 1 center + 5 inner ring + 10 outer ring
    # This is a natural 16-point packing
    center = np.array([[0.0, 0.0]])
    for r1, r2, phase1, phase2 in [
        (0.5, 1.2, 0.0, 0.0),
        (0.5, 1.2, 0.0, np.pi/10),
        (0.6, 1.3, 0.0, np.pi/10),
        (0.55, 1.25, np.pi/10, 0.0),
        (0.5, 1.1, 0.0, np.pi/5),
        (0.6, 1.4, 0.0, np.pi/10),
    ]:
        angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False) + phase1
        angles_out = np.linspace(0, 2*np.pi, 10, endpoint=False) + phase2
        inner = np.column_stack([r1 * np.cos(angles_in), r1 * np.sin(angles_in)])
        outer = np.column_stack([r2 * np.cos(angles_out), r2 * np.sin(angles_out)])
        candidates.append(np.vstack([center, inner, outer]))
    
    # 2. Hexagonal lattice subset (centered)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    candidates.append(hex_points[sorted_indices][:16])
    # Jittered hex
    for sigma in [0.05, 0.1]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 3. 4x4 grid (centered)
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 4. Concentric rings: 8 inner + 8 outer
    for r1, r2, phase in [(0.5, 1.5, 0), (0.6, 1.6, np.pi/8), (0.55, 1.4, np.pi/4)]:
        angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase
        angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase + np.pi/8
        inner = np.column_stack([r1 * np.cos(angles_in), r1 * np.sin(angles_in)])
        outer = np.column_stack([r2 * np.cos(angles_out), r2 * np.sin(angles_out)])
        candidates.append(np.vstack([inner, outer]))
    
    # 5. Random configurations
    for _ in range(8):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # 6. Random on rings
    for _ in range(4):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 7. Square lattice with radial perturbation
    base_grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    r = np.sqrt(np.sum(base_grid**2, axis=1))
    for factor in [0.2, 0.3, 0.4]:
        scaled = base_grid * (1.0 + factor * np.sqrt(r)[:, np.newaxis])
        candidates.append(scaled)
    
    return candidates

def _simulated_annealing(params, start_temp=1.0, end_temp=0.01, n_iter=5000, 
                         step_size=0.1, rng=None, deadline=None):
    """
    Simulated annealing on the exact objective in reduced parameter space.
    Uses Gaussian perturbations with adaptive step size.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_params = params.copy()
    current_score = _exact_objective(current_params)
    best_params = current_params.copy()
    best_score = current_score
    
    n_params = len(current_params)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Adaptive temperature
        alpha = i / n_iter
        temp = start_temp * (end_temp / start_temp) ** alpha
        
        # Generate perturbation - adaptive step size
        # Larger steps early, smaller steps late
        adapt_step = step_size * (1.0 - 0.5 * alpha)
        delta = rng.normal(0, adapt_step, n_params)
        new_params = current_params + delta
        
        new_score = _exact_objective(new_params)
        
        # Accept if better or with probability exp(-delta_E / T)
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()
        else:
            delta_e = new_score - current_score
            if delta_e < 1e-10:
                current_params = new_params
                current_score = new_score
            else:
                # Accept with probability exp(-delta_e / temp)
                # Use log to avoid underflow
                log_prob = -delta_e / (temp + 1e-15)
                if log_prob > -50:  # avoid underflow
                    if np.log(rng.random()) < log_prob:
                        current_params = new_params
                        current_score = new_score
    
    return best_params, best_score

def _local_optimize(params, deadline=None):
    """
    Local optimization using Powell's method (derivative-free, robust to nonsmoothness).
    """
    if deadline is not None and time.time() > deadline:
        return params, _exact_objective(params)
    
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Powell's method - derivative-free, good for nonsmooth objectives
    try:
        res = minimize(
            _exact_objective,
            best_params,
            method='Powell',
            options={
                'maxiter': 2000,
                'xtol': 1e-10,
                'ftol': 1e-10,
                'maxfev': 5000,
            }
        )
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Nelder-Mead as secondary
    if deadline is None or time.time() < deadline:
        try:
            res_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    candidates = _get_initial_candidates()
    rng = np.random.default_rng(123)
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: SA + Powell from diverse structured starts
    for P0 in candidates:
        if time.time() > deadline - 30:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            # SA for global exploration
            sa_params, sa_score = _simulated_annealing(
                params, start_temp=0.5, end_temp=0.01, n_iter=3000,
                step_size=0.15, rng=rng, deadline=deadline - 25
            )
            
            # Powell for local polishing
            p_opt, score = _local_optimize(sa_params, deadline=deadline - 20)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: More random restarts with SA
    n_random_restarts = 30
    for i in range(n_random_restarts):
        if time.time() > deadline - 30:
            break
        try:
            # Random initial in reduced space
            P0 = rng.uniform(-2, 2, (14, 2))
            params = P0.flatten()
            
            # SA with moderate iterations
            sa_params, sa_score = _simulated_annealing(
                params, start_temp=0.3, end_temp=0.01, n_iter=2000,
                step_size=0.1, rng=rng, deadline=deadline - 25
            )
            
            # Powell polish
            p_opt, score = _local_optimize(sa_params, deadline=deadline - 20)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish on best solution
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _exact_objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 5000,
                    'xtol': 1e-12,
                    'ftol': 1e-12,
                    'maxfev': 20000,
                }
            )
            exact_s = _exact_objective(res_final.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final.x.copy()
        except Exception:
            pass
        
        # One more Nelder-Mead polish
        if time.time() < deadline - 3:
            try:
                res_nm = minimize(
                    _exact_objective,
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                exact_s = _exact_objective(res_nm.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    # Reconstruct final points
    if best_params is None:
        # Fallback
        best_params = _prepare_params(candidates[0])
        if best_params is None:
            return candidates[0]
    
    P_final = _reconstruct(best_params)
    return P_final

# EVOLVE-BLOCK-END
