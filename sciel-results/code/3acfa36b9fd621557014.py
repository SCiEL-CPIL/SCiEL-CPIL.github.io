import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

def _soft_max_logsumexp(x, T=1.0):
    """Stable soft max via log-sum-exp."""
    x_scaled = x / T
    m = np.max(x_scaled)
    if m < -1000:
        return m * T
    return T * (m + np.log(np.sum(np.exp(x_scaled - m))))

def _soft_min_logsumexp(x, T=1.0):
    """Stable soft min via log-sum-exp on -x."""
    return -_soft_max_logsumexp(-x, T)

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    # P is (N, 2)
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

def _soft_spread_objective(params, T=1.0):
    """
    Smooth surrogate objective using soft max/min.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    soft_max = _soft_max_logsumexp(d_sq, T)
    soft_min = _soft_min_logsumexp(d_sq, T)
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min

def _epigraph_objective(params, T=1.0):
    """
    Epigraph formulation: minimize t subject to soft_max(d_sq) <= t * soft_min(d_sq).
    This is reformulated as minimizing t + penalty * max(0, soft_max - t*soft_min).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    soft_max = _soft_max_logsumexp(d_sq, T)
    soft_min = _soft_min_logsumexp(d_sq, T)
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min

def _soft_spread_gradient(params, T=1.0):
    """
    Analytical gradient of the soft spread objective.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    # Compute all pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq_full = np.sum(diff**2, axis=2)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, 1)
    d_sq = d_sq_full[iu]
    
    # Soft max weights: softmax(d_sq / T)
    d_scaled = d_sq / T
    m = np.max(d_scaled)
    exp_vals = np.exp(d_scaled - m)
    softmax_weights = exp_vals / np.sum(exp_vals)
    
    # Soft min weights: softmax(-d_sq / T)
    neg_d_scaled = -d_sq / T
    m_neg = np.max(neg_d_scaled)
    exp_vals_neg = np.exp(neg_d_scaled - m_neg)
    softmin_weights = exp_vals_neg / np.sum(exp_vals_neg)
    
    soft_max = T * (m + np.log(np.sum(exp_vals)))
    soft_min = -T * (m_neg + np.log(np.sum(exp_vals_neg)))
    
    if soft_min < 1e-8:
        return np.zeros_like(params)
    
    # d(soft_max)/d(d_sq) = softmax_weights / T
    # d(soft_min)/d(d_sq) = -softmin_weights / T
    
    # Objective = soft_max / soft_min
    # d(obj)/d(d_sq) = (d(soft_max)/d(d_sq) * soft_min - soft_max * d(soft_min)/d(d_sq)) / soft_min^2
    #                = (softmax_weights / T * soft_min + soft_max * softmin_weights / T) / soft_min^2
    #                = (softmax_weights * soft_min + soft_max * softmin_weights) / (T * soft_min^2)
    
    grad_d_sq = (softmax_weights * soft_min + soft_max * softmin_weights) / (T * soft_min**2)
    
    # Now compute gradient w.r.t. P
    # d(d_sq_ij)/d(P_k) = 2 * (P_k - P_j) if k == i, 2 * (P_k - P_i) if k == j, 0 otherwise
    # For upper triangle, i < j
    
    grad_P = np.zeros((N, 2))
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        g = grad_d_sq[idx]
        # d(d_sq_ij)/d(P_i) = 2*(P_i - P_j)
        grad_P[i] += 2 * g * (P[i] - P[j])
        # d(d_sq_ij)/d(P_j) = 2*(P_j - P_i)
        grad_P[j] += 2 * g * (P[j] - P[i])
    
    # Gradient w.r.t. params (which are P[2:].flatten())
    grad_params = grad_P[2:].flatten()
    
    return grad_params

def _exact_objective(params):
    """
    Exact objective for local refinement.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _coordinate_descent_polish(params, max_iters=200, tol=1e-12, deadline=None):
    """
    Coordinate descent on the exact nonsmooth objective.
    Cycles through each coordinate, doing a 1D line search.
    """
    current = params.copy()
    current_score = _exact_objective(current)
    
    n_params = len(current)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for coord in range(n_params):
            if deadline is not None and time.time() > deadline:
                break
            
            # Golden section search on coordinate `coord`
            best_val = current[coord]
            best_s = current_score
            
            # Try expanding the search interval
            step = 0.01
            for _ in range(8):
                # Try +step
                trial = current.copy()
                trial[coord] = best_val + step
                s_plus = _exact_objective(trial)
                if s_plus < best_s:
                    best_s = s_plus
                    best_val = trial[coord]
                    improved = True
                else:
                    break
                step *= 0.5
            
            step = 0.01
            for _ in range(8):
                # Try -step
                trial = current.copy()
                trial[coord] = best_val - step
                s_minus = _exact_objective(trial)
                if s_minus < best_s:
                    best_s = s_minus
                    best_val = trial[coord]
                    improved = True
                else:
                    break
                step *= 0.5
            
            current[coord] = best_val
        
        if not improved:
            break
    
    final_score = _exact_objective(current)
    return current, final_score

def _get_initial_candidates():
    """
    Generate diverse initial configurations.
    Returns list of (N, 2) arrays.
    """
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Hexagonal Lattice Subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    if len(outer_indices) >= 3:
        # Remove 3 outer points to get 16
        for trial in range(8):
            if trial == 0:
                remove_idx = list(outer_indices[:3])
            else:
                remove_idx = list(rng.choice(outer_indices, 3, replace=False))
            keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
            pts_hex = hex_points[keep_idx]
            candidates.append(pts_hex)

    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles (8 inner, 8 outer) with optimized radii
    for r_inner, r_outer in [(0.5, 1.5), (0.6, 1.4), (0.7, 1.3), (1.0, 2.0)]:
        pts_circ = []
        for k in range(8):
            ang = 2 * np.pi * k / 8
            pts_circ.append([r_outer * np.cos(ang), r_outer * np.sin(ang)])
        for k in range(8):
            ang = 2 * np.pi * (k + 0.5) / 8
            pts_circ.append([r_inner * np.cos(ang), r_inner * np.sin(ang)])
        candidates.append(np.array(pts_circ))
    
    # 4. Random Perturbed Grid
    perturbed = grid + rng.normal(0, 0.15, grid.shape)
    candidates.append(perturbed)
    
    # 5. Random Gaussian Cluster
    gauss = rng.normal(0, 1, (N, 2))
    candidates.append(gauss)
    
    # 6. Ring of 16
    pts_ring = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_ring.append([2 * np.cos(ang), 2 * np.sin(ang)])
    candidates.append(np.array(pts_ring))
    
    # 7. Three concentric rings: 6, 6, 4
    for r1, r2, r3 in [(0.5, 1.0, 1.5), (0.6, 1.1, 1.6), (0.4, 0.9, 1.4)]:
        pts_3rings = []
        for k in range(6):
            ang = 2 * np.pi * k / 6
            pts_3rings.append([r1 * np.cos(ang), r1 * np.sin(ang)])
        for k in range(6):
            ang = 2 * np.pi * (k + 0.5) / 6
            pts_3rings.append([r2 * np.cos(ang), r2 * np.sin(ang)])
        for k in range(4):
            ang = 2 * np.pi * k / 4
            pts_3rings.append([r3 * np.cos(ang), r3 * np.sin(ang)])
        candidates.append(np.array(pts_3rings))
    
    # 8. Square grid with alternating offsets (quasi-crystal-like)
    sq_pts = []
    for i in range(4):
        for j in range(4):
            x = i - 1.5
            y = j - 1.5
            if (i + j) % 2 == 1:
                x += 0.1
                y += 0.1
            sq_pts.append([x, y])
    candidates.append(np.array(sq_pts))
    
    # 9. Fibonacci spiral
    fib_pts = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for k in range(16):
        r = np.sqrt(k + 1) * 0.6
        ang = k * golden_angle
        fib_pts.append([r * np.cos(ang), r * np.sin(ang)])
    candidates.append(np.array(fib_pts))
    
    # 10. Random diverse points
    for seed in [123, 456, 789]:
        rng_temp = np.random.default_rng(seed)
        rand_pts = rng_temp.uniform(-2, 2, (N, 2))
        candidates.append(rand_pts)
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/ translating/scaling the whole set so that pair becomes (0,0) and (1,0).
    Returns the optimized parameter vector (coords of P[2:] to P[15]) and the permutation used.
    """
    # Find min distance pair
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    p, q = np.unravel_index(min_idx, (N, N))
    if p > q:
        p, q = q, p
    
    p0 = P[p]
    p1 = P[q]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        # Degenerate, return original flattened (bad)
        return None, None
        
    # Rotation matrix to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (P - p0) @ R_mat.T / norm_v
    
    # Reorder points so that p is at index 0 and q is at index 1
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # Parameters are the coordinates of points 2..15
    init_params = transformed[2:].flatten()
    
    return init_params, perm

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = _get_initial_candidates()
    
    # Pre-compute best initial candidate score for fallback
    best_init_score = float('inf')
    best_init_P = None
    for P0 in candidates:
        s = _score(P0)
        if s < best_init_score:
            best_init_score = s
            best_init_P = P0.copy()
    
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Differential Evolution for global exploration
    # Define bounds for the 28 free coordinates (points 2..15)
    # Points 0 and 1 are fixed at (0,0) and (1,0)
    # Use reasonable bounds based on the problem scale
    bounds = [(-5.0, 5.0)] * (2 * (N - 2))
    
    try:
        de_result = differential_evolution(
            _exact_objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,  # We'll do our own polishing
            init='sobol',
            workers=1,
            update_interval=10,
            callback=None
        )
        
        if de_result.success or de_result.fun < best_score:
            best_score = de_result.fun
            best_params = de_result.x.copy()
            
    except Exception:
        pass
    
    # Also try DE from the best initial candidate neighborhood
    if best_init_P is not None and time.time() < deadline - 30:
        try:
            init_params, perm = _prepare_for_optimization(best_init_P)
            if init_params is not None:
                # Tighter bounds around the initial solution
                local_bounds = []
                for i, v in enumerate(init_params):
                    local_bounds.append((v - 2.0, v + 2.0))
                
                de_local = differential_evolution(
                    _exact_objective,
                    bounds=local_bounds,
                    seed=123,
                    maxiter=50,
                    popsize=12,
                    tol=1e-7,
                    mutation=(0.2, 0.8),
                    recombination=0.7,
                    polish=False,
                    init='latinhypercube'
                )
                
                if de_local.fun < best_score:
                    best_score = de_local.fun
                    best_params = de_local.x.copy()
                    
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement with soft objective (analytical gradients)
    if best_params is not None and time.time() < deadline - 20:
        try:
            # Use soft objective with moderate temperature for smooth landscape
            res_lbf = minimize(
                lambda x: _soft_spread_objective(x, 0.5),
                lambda x: _soft_spread_gradient(x, 0.5),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            P_lbf = np.zeros((N, 2))
            P_lbf[0] = [0.0, 0.0]
            P_lbf[1] = [1.0, 0.0]
            P_lbf[2:] = res_lbf.x.reshape(-1, 2)
            
            s_lbf = _score(P_lbf)
            if s_lbf < best_score:
                best_score = s_lbf
                best_params = res_lbf.x.copy()
                
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish on the exact nonsmooth objective
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            P_nm = np.zeros((N, 2))
            P_nm[0] = [0.0, 0.0]
            P_nm[1] = [1.0, 0.0]
            P_nm[2:] = res_nm.x.reshape(-1, 2)
            
            nm_score = _score(P_nm)
            
            if nm_score < best_score:
                best_score = nm_score
                best_params = res_nm.x.copy()
                
        except Exception:
            pass
    
    # Phase 4: Coordinate descent polish on the exact objective
    if best_params is not None and time.time() < deadline - 3:
        try:
            cd_params, cd_score = _coordinate_descent_polish(
                best_params, max_iters=100, tol=1e-12, deadline=deadline - 2
            )
            if cd_score < best_score:
                best_score = cd_score
                best_params = cd_params.copy()
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead with tighter tolerances if time remains
    if best_params is not None and time.time() < deadline - 1:
        try:
            res_final_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            P_final_nm = np.zeros((N, 2))
            P_final_nm[0] = [0.0, 0.0]
            P_final_nm[1] = [1.0, 0.0]
            P_final_nm[2:] = res_final_nm.x.reshape(-1, 2)
            
            final_nm_score = _score(P_final_nm)
            
            if final_nm_score < best_score:
                best_score = final_nm_score
                best_params = res_final_nm.x.copy()
                
        except Exception:
            pass

    # Construct final points
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Check validity
        if _score(final_pts) > 1e6:
             # Fallback to best initial candidate
             return best_init_P
             
        return final_pts
    else:
        # Fallback to best initial candidate
        return best_init_P

# EVOLVE-BLOCK-END
