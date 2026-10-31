import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

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
    
    # Rotation to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform: translate so p0 is origin, rotate, scale so p1 is at (1,0)
    transformed = (P - p0) @ R.T / norm_v
    
    # Reorder: i_min -> 0, j_min -> 1, rest in original order
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

def _soft_objective(params, beta=50.0):
    """
    Soft min/max surrogate for gradient-based optimization.
    Uses log-sum-exp for numerical stability.
    """
    P = _reconstruct(params)
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max via log-sum-exp
    max_d_sq = np.max(d_sq)
    soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min
    min_d_sq = np.min(d_sq)
    soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min

def _epigraph_objective(params, t, beta=100.0):
    """
    Epigraph formulation: minimize t subject to soft constraints.
    t >= soft_max(d_sq) and t >= 1/soft_min(d_sq) approximately.
    This makes the problem more amenable to gradient-based methods.
    """
    P = _reconstruct(params)
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    max_d_sq = np.max(d_sq)
    min_d_sq = np.min(d_sq)
    
    # Soft max of distances squared
    soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min of distances squared  
    soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min < 1e-8:
        return 1e10
    
    ratio_sq = soft_max / soft_min
    
    # Penalty for t being less than ratio_sq
    if t < ratio_sq:
        return ratio_sq + 10.0 * (ratio_sq - t)**2
    return t

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Hexagonal lattice subset
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
    for sigma in [0.05, 0.1, 0.2]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 2. 4x4 grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 3. Concentric circles (8 inner + 8 outer)
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    
    # 4. Random uniform
    for _ in range(5):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # 5. Random on ring
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Square-root radial grid: 4x4 grid with radial scaling
    # Each point displaced radially from center by factor proportional to sqrt(index)
    base_grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    # Compute radial distance from center for each grid point
    r = np.sqrt(np.sum(base_grid**2, axis=1))
    # Apply square-root radial scaling
    scaled = base_grid * (1.0 + 0.3 * np.sqrt(r)[:, np.newaxis])
    candidates.append(scaled)
    
    # 7. Optimized concentric rings
    for r1, r2, phase in [(0.4, 1.2, 0), (0.5, 1.4, np.pi/8), (0.6, 1.6, np.pi/4)]:
        angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase
        angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase + np.pi/8
        inner = np.column_stack([r1 * np.cos(angles_in), r1 * np.sin(angles_in)])
        outer = np.column_stack([r2 * np.cos(angles_out), r2 * np.sin(angles_out)])
        candidates.append(np.vstack([inner, outer]))
    
    return candidates

def _local_optimize_params(params, maxiter_nm=500, maxiter_lbfgs=500, time_budget=10.0):
    """
    Multi-phase local optimization in reduced parameter space:
    1. Nelder-Mead on exact objective
    2. L-BFGS-B on soft objective for gradient-based polish
    3. Epigraph formulation with L-BFGS-B for final refinement
    Returns best params and exact score.
    """
    start_time = time.time()
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Phase 1: Nelder-Mead
    try:
        res_nm = minimize(
            _exact_objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_params = res_nm.x.copy()
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.4:
        return best_params, best_score
    
    # Phase 2: L-BFGS-B on soft objective
    try:
        res_lbfgs = minimize(
            lambda p: _soft_objective(p, beta=50.0),
            best_params,
            method='L-BFGS-B',
            options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        exact_s = _exact_objective(res_lbfgs.x)
        if exact_s < best_score:
            best_score = exact_s
            best_params = res_lbfgs.x.copy()
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.7:
        return best_params, best_score
    
    # Phase 3: Epigraph formulation with higher beta
    try:
        initial_t = best_score * 1.01
        res_epigraph = minimize(
            lambda x: _epigraph_objective(x[:28], x[28], beta=100.0),
            np.concatenate([best_params, [initial_t]]),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        epigraph_params = res_epigraph.x[:28]
        exact_s = _exact_objective(epigraph_params)
        if exact_s < best_score:
            best_score = exact_s
            best_params = epigraph_params.copy()
    except Exception:
        pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    candidates = _get_initial_candidates()
    rng = np.random.default_rng(123)
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Local optimization from diverse starts
    for P0 in candidates:
        if time.time() > deadline - 30:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            p_opt, score = _local_optimize_params(params, maxiter_nm=200, maxiter_lbfgs=200, time_budget=8.0)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: Particle-equilibrium inspired optimization
    # Model points as repelling particles in a variable-radius container
    if time.time() < deadline - 30:
        try:
            # Start from current best
            P_current = _reconstruct(best_params)
            
            # Try different container radii and repulsion strengths
            for R_container in [1.5, 1.8, 2.0, 2.2, 2.5]:
                if time.time() > deadline - 25:
                    break
                
                # Initialize with points scaled to fit in container
                P_init = P_current.copy()
                # Scale to fit in circle of radius R_container
                max_r = np.max(np.linalg.norm(P_init, axis=1))
                if max_r > 0:
                    P_init = P_init * (R_container * 0.8 / max_r)
                
                params_init = _prepare_params(P_init)
                if params_init is None:
                    continue
                
                # Optimize with emphasis on uniform spacing
                p_opt, score = _local_optimize_params(params_init, maxiter_nm=150, maxiter_lbfgs=150, time_budget=5.0)
                if score < best_score:
                    best_score = score
                    best_params = p_opt
        except Exception:
            pass
    
    # Phase 3: Random restarts in reduced space
    while time.time() < deadline - 25:
        try:
            # Generate random params in reduced space
            P0 = rng.uniform(-2, 2, (14, 2))
            params = P0.flatten()
            
            p_opt, score = _local_optimize_params(params, maxiter_nm=150, maxiter_lbfgs=150, time_budget=6.0)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_params is not None and time.time() < deadline - 10:
        # Polish with soft objective at very high beta
        try:
            res_final1 = minimize(
                lambda p: _soft_objective(p, beta=500.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            exact_s = _exact_objective(res_final1.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final1.x.copy()
        except Exception:
            pass
        
        # Additional epigraph polish
        if time.time() < deadline - 5:
            try:
                initial_t = best_score * 1.001
                res_final2 = minimize(
                    lambda x: _epigraph_objective(x[:28], x[28], beta=200.0),
                    np.concatenate([best_params, [initial_t]]),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
                )
                epigraph_params = res_final2.x[:28]
                exact_s = _exact_objective(epigraph_params)
                if exact_s < best_score:
                    best_score = exact_s
                    best_params = epigraph_params.copy()
            except Exception:
                pass
    
    # Reconstruct final points
    if best_params is None:
        # Fallback
        best_params = _prepare_params(candidates[0])
        if best_params is None:
            return candidates[0]
    
    P_final = _reconstruct(best_params)
    
    # Final validation: ensure all points are distinct
    d_sq_final = _pairwise_dists_sq(P_final)
    if np.min(d_sq_final) < 1e-10:
        # Add tiny perturbation to ensure distinctness
        rng_final = np.random.default_rng(999)
        P_final = P_final + rng_final.normal(0, 1e-8, (N, 2))
    
    return P_final

# EVOLVE-BLOCK-END
