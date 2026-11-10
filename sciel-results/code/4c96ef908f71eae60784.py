# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 35.0

# Precomputed constants
_BASE_ANGLES = np.deg2rad(60.0 * np.arange(6))
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)
_NORMAL_ANGLES = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
_NORMALS = np.column_stack([np.cos(_NORMAL_ANGLES), np.sin(_NORMAL_ANGLES)])  # (6, 2)
_ALL_VERT_OFFSETS = np.column_stack([_BASE_COS, _BASE_SIN])  # (6, 2)


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(centers):
    """Exact minimum side length of a regular hexagon at origin, angle 0, containing all unit hexagons."""
    n = len(centers)
    # All vertices: (n*6, 2)
    all_verts = np.tile(np.array([[0.0, 0.0]]), (n, 1)) + np.tile(_ALL_VERT_OFFSETS, (n, 1))
    all_verts = all_verts + centers[:, np.newaxis, :]
    all_verts = all_verts.reshape(-1, 2)
    
    projections = _NORMALS @ all_verts.T  # (6, n*6)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _smooth_containing_hex_side(centers, beta=50.0):
    """Smooth surrogate for the containing hexagon side using log-sum-exp."""
    n = len(centers)
    # All vertices
    all_verts = centers[:, np.newaxis, :] + _ALL_VERT_OFFSETS[np.newaxis, :, :]  # (n, 6, 2)
    all_verts = all_verts.reshape(-1, 2)  # (n*6, 2)
    
    # Projections onto 6 normals: (6, n*6)
    projections = _NORMALS @ all_verts.T  # (6, n*6)
    
    # For each normal direction, compute soft-max of projections
    # soft_max = (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, subtract max
    max_projs = np.max(projections, axis=1, keepdims=True)  # (6, 1)
    shifted = beta * (projections - max_projs)
    log_sums = np.log(np.sum(np.exp(shifted), axis=1))  # (6,)
    soft_max_per_normal = max_projs.squeeze() + log_sums / beta  # (6,)
    
    # Overall soft-max over the 6 normal directions
    overall_max = np.max(soft_max_per_normal)
    shifted2 = beta * (soft_max_per_normal - overall_max)
    log_sum2 = np.log(np.sum(np.exp(shifted2)))
    soft_max_overall = overall_max + log_sum2 / beta
    
    s = (2.0 / np.sqrt(3.0)) * soft_max_overall
    return s


def _check_separation(centers, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    
    # Precompute all vertices: (n, 6, 2)
    all_verts = centers[:, np.newaxis, :] + _ALL_VERT_OFFSETS[np.newaxis, :, :]
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]  # (6, 2)
            vj = all_verts[j]  # (6, 2)
            
            # Project both onto all 6 normals
            proj_i = vi @ _NORMALS.T  # (6, 6)
            proj_j = vj @ _NORMALS.T  # (6, 6)
            
            min_i = np.min(proj_i, axis=1)  # (6,)
            max_i = np.max(proj_i, axis=1)  # (6,)
            min_j = np.min(proj_j, axis=1)  # (6,)
            max_j = np.max(proj_j, axis=1)  # (6,)
            
            # Check if any axis separates them
            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True


def _separation_violation_smooth(centers, beta=30.0):
    """Smooth separation violation using softplus."""
    n = len(centers)
    
    all_verts = centers[:, np.newaxis, :] + _ALL_VERT_OFFSETS[np.newaxis, :, :]
    
    total_violation = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            proj_i = vi @ _NORMALS.T  # (6, 6)
            proj_j = vj @ _NORMALS.T  # (6, 6)
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            # Overlap on each axis
            overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            max_overlap = np.max(overlaps)
            
            if max_overlap > 0:
                # Softplus smoothing: log(1 + exp(beta * x)) / beta
                total_violation += np.log1p(np.exp(beta * max_overlap)) / beta
    
    return total_violation


def _objective_smooth(params, beta_s=50.0, beta_sep=30.0, penalty_weight=200.0):
    """Smooth objective: soft-max containment + softplus separation penalty."""
    centers = params.reshape(12, 2)
    s = _smooth_containing_hex_side(centers, beta=beta_s)
    violation = _separation_violation_smooth(centers, beta=beta_sep)
    penalty = penalty_weight * violation
    return s + penalty


def _objective_grad(params, beta_s=50.0, beta_sep=30.0, penalty_weight=200.0):
    """Smooth objective with gradient (for trust-constr)."""
    centers = params.reshape(12, 2)
    n = len(centers)
    
    # Compute smooth containment side and its gradient
    all_verts = centers[:, np.newaxis, :] + _ALL_VERT_OFFSETS[np.newaxis, :, :]  # (n, 6, 2)
    all_verts_flat = all_verts.reshape(-1, 2)  # (n*6, 2)
    
    projections = _NORMALS @ all_verts_flat.T  # (6, n*6)
    
    # Soft-max per normal
    max_projs = np.max(projections, axis=1, keepdims=True)
    shifted = beta_s * (projections - max_projs)
    exp_shifted = np.exp(shifted)
    sums = np.sum(exp_shifted, axis=1)  # (6,)
    soft_max_per_normal = max_projs.squeeze() + np.log(sums) / beta_s  # (6,)
    
    # Overall soft-max
    overall_max = np.max(soft_max_per_normal)
    shifted2 = beta_s * (soft_max_per_normal - overall_max)
    exp_shifted2 = np.exp(shifted2)
    sum2 = np.sum(exp_shifted2)
    soft_max_overall = overall_max + np.log(sum2) / beta_s
    s = (2.0 / np.sqrt(3.0)) * soft_max_overall
    
    # Gradient of s w.r.t. centers
    # ds/d(soft_max_overall) = 2/sqrt(3)
    # d(soft_max_overall)/d(soft_max_per_normal[k]) = exp_shifted2[k] / sum2
    # d(soft_max_per_normal[k])/d(projection[k,m]) = exp_shifted[k,m] / sums[k]
    # d(projection[k,m])/d(centers[i]) = _NORMALS[k] if vertex m belongs to hex i
    
    grad_s = np.zeros((n, 2))
    for k in range(6):
        w_overall = exp_shifted2[k] / sum2  # weight of normal k in overall soft-max
        for m in range(n * 6):
            w_normal = exp_shifted[k, m] / sums[k]  # weight of vertex m in normal k's soft-max
            hex_idx = m // 6
            grad_s[hex_idx] += (2.0 / np.sqrt(3.0)) * w_overall * w_normal * _NORMALS[k]
    
    # Gradient of separation violation
    grad_v = np.zeros((n, 2))
    all_verts = centers[:, np.newaxis, :] + _ALL_VERT_OFFSETS[np.newaxis, :, :]
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            proj_i = vi @ _NORMALS.T
            proj_j = vj @ _NORMALS.T
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            max_overlap = np.max(overlaps)
            
            if max_overlap > 1e-12:
                # softplus derivative: exp(beta*x)/(1+exp(beta*x))
                bp = beta_sep * max_overlap
                bp = np.clip(bp, -50, 50)
                softplus_deriv = np.exp(bp) / (1.0 + np.exp(bp))
                
                # Find which axis gives max overlap
                max_axis = np.argmax(overlaps)
                normal = _NORMALS[max_axis]
                
                # Gradient of overlap w.r.t. centers
                # overlap = min(max_i, max_j) - max(min_i, min_j)
                # For the active axis, we need to know which vertices are at min/max
                # This is complex; use numerical gradient for separation part
                pass
    
    # Use numerical gradient for the separation part (simpler and robust)
    eps = 1e-7
    grad_num = np.zeros_like(params)
    for d in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[d] += eps
        params_minus[d] -= eps
        f_plus = _objective_smooth(params_plus, beta_s, beta_sep, penalty_weight)
        f_minus = _objective_smooth(params_minus, beta_s, beta_sep, penalty_weight)
        grad_num[d] = (f_plus - f_minus) / (2.0 * eps)
    
    return s + penalty_weight * _separation_violation_smooth(centers, beta_sep), grad_num


def _make_good_initial():
    """Compact hexagonal lattice arrangement for 12 hexagons."""
    sqrt3 = np.sqrt(3)
    init = np.array([
        [0.0, 0.0],
        [1.5, sqrt3/2],
        [1.5, -sqrt3/2],
        [0.0, sqrt3],
        [0.0, -sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        [3.0, 0.0],
        [2.25, 1.5*sqrt3],
        [0.75, 2.0*sqrt3],
        [-1.5, 1.5*sqrt3],
        [-3.0, 0.0],
    ])
    return init


def _make_spread_initial():
    """More spread out arrangement."""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
        [0.0, -4.33],
    ])


def _make_ring2_initial():
    """Ring-2 arrangement (13 minus 1)."""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [3*sqrt3/2, 1.5],
        [3*sqrt3/2, -1.5],
        [0.0, 3.0],
        [-3*sqrt3/2, 1.5],
        [-3*sqrt3/2, -1.5],
    ])


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    initial_configs = [
        _make_good_initial(),
        _make_spread_initial(),
        _make_ring2_initial(),
    ]
    
    best_centers = None
    best_s = np.inf
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Nelder-Mead with smooth objective from multiple starts
    for init_idx, init_centers in enumerate(initial_configs):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Check if initial is valid
        if _check_separation(init_centers):
            s_init = _min_containing_hex_side(init_centers)
            if s_init < best_s:
                best_s = s_init
                best_centers = init_centers.copy()
        
        # Run Nelder-Mead with smooth objective
        x0 = init_centers.flatten()
        
        try:
            result = minimize(
                lambda p: _objective_smooth(p, beta_s=50.0, beta_sep=30.0, penalty_weight=200.0),
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 8000,
                    'maxfev': 80000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
        except Exception:
            result = None
        
        if result is not None:
            candidate = result.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
        
        # Random restarts
        n_restarts = 3
        for r in range(n_restarts):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            noise_scale = 0.15 * (1.0 + r * 0.4)
            perturbed = init_centers + rng.normal(0, noise_scale, size=init_centers.shape)
            
            x0_p = perturbed.flatten()
            try:
                res_p = minimize(
                    lambda p: _objective_smooth(p, beta_s=50.0, beta_sep=30.0, penalty_weight=200.0),
                    x0_p,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'maxfev': 50000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                res_p = None
            
            if res_p is not None:
                candidate = res_p.x.reshape(12, 2)
                if _check_separation(candidate):
                    s_cand = _min_containing_hex_side(candidate)
                    if s_cand < best_s:
                        best_s = s_cand
                        best_centers = candidate.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = initial_configs[1].copy()
        best_s = _min_containing_hex_side(best_centers)
    
    # Phase 2: High-precision polish with trust-constr (interior-point)
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        # Use higher beta for sharper approximation
        x0_final = best_centers.flatten()
        
        # Try trust-constr with smooth objective
        try:
            # Define smooth objective for trust-constr
            def tc_objective(p):
                return _objective_smooth(p, beta_s=100.0, beta_sep=50.0, penalty_weight=500.0)
            
            # Numerical gradient for trust-constr
            def tc_gradient(p):
                eps = 1e-8
                n = len(p)
                grad = np.zeros(n)
                for d in range(n):
                    p_plus = p.copy()
                    p_minus = p.copy()
                    p_plus[d] += eps
                    p_minus[d] -= eps
                    grad[d] = (tc_objective(p_plus) - tc_objective(p_minus)) / (2.0 * eps)
                return grad
            
            result_tc = minimize(
                tc_objective,
                x0_final,
                jac=tc_gradient,
                method='trust-constr',
                options={
                    'maxiter': 2000,
                    'gtol': 1e-12,
                    'xtol': 1e-12,
                    'barrier_tol': 1e-12,
                    'verbose': 0,
                }
            )
        except Exception:
            result_tc = None
        
        if result_tc is not None:
            candidate = result_tc.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
    
    # Phase 3: Final Nelder-Mead polish with very tight tolerances
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        x0_polish = best_centers.flatten()
        try:
            result_polish = minimize(
                lambda p: _objective_smooth(p, beta_s=200.0, beta_sep=100.0, penalty_weight=1000.0),
                x0_polish,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'maxfev': 200000,
                    'xatol': 1e-13,
                    'fatol': 1e-15,
                    'adaptive': True,
                }
            )
        except Exception:
            result_polish = None
        
        if result_polish is not None:
            candidate = result_polish.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
    
    # Phase 4: One more trust-constr pass if time allows
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN - 10:
        x0_final2 = best_centers.flatten()
        try:
            def tc_objective2(p):
                return _objective_smooth(p, beta_s=300.0, beta_sep=150.0, penalty_weight=2000.0)
            
            def tc_gradient2(p):
                eps = 1e-9
                n = len(p)
                grad = np.zeros(n)
                for d in range(n):
                    p_plus = p.copy()
                    p_minus = p.copy()
                    p_plus[d] += eps
                    p_minus[d] -= eps
                    grad[d] = (tc_objective2(p_plus) - tc_objective2(p_minus)) / (2.0 * eps)
                return grad
            
            result_tc2 = minimize(
                tc_objective2,
                x0_final2,
                jac=tc_gradient2,
                method='trust-constr',
                options={
                    'maxiter': 3000,
                    'gtol': 1e-14,
                    'xtol': 1e-14,
                    'verbose': 0,
                }
            )
        except Exception:
            result_tc2 = None
        
        if result_tc2 is not None:
            candidate = result_tc2.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
