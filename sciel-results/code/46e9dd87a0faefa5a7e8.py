# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(centers):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0), angle 0,
    that contains all unit hexagons with given centers.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # All vertices for all hexagons: (n, 6, 2)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i, :, 0] = centers[i, 0] + base_cos
        verts[i, :, 1] = centers[i, 1] + base_sin

    # Outward normals at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    verts_flat = verts.reshape(-1, 2)  # (36, 2)
    projections = normals @ verts_flat.T  # (6, 36)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)  # scalar
    
    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _make_good_initial():
    """
    Create a good initial hexagonal lattice arrangement.
    For 12 hexagons, use a compact hexagonal cluster.
    """
    # For flat-top hexagons with side 1:
    # Width = 2, Height = sqrt(3)
    # Horizontal spacing between columns: 1.5
    # Vertical spacing: sqrt(3)
    # Odd columns shifted up by sqrt(3)/2
    
    # Start with a compact arrangement
    init = np.array([
        [0.0, 0.0],           # center
        [1.5, 0.866],         # ring 1
        [1.5, -0.866],
        [0.0, 1.732],
        [0.0, -1.732],
        [-1.5, 0.866],
        [-1.5, -0.866],
        # ring 2 (partial)
        [3.0, 0.0],
        [2.25, 2.598],
        [0.75, 3.464],
        [-1.5, 2.598],
        [-3.0, 0.0],
    ])
    
    return init


def _check_separation(centers, tol=1e-6):
    """
    Check if all pairs of unit hexagons are separated using SAT.
    Returns True if all pairs are separated.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = np.column_stack([centers[i, 0] + base_cos, centers[i, 1] + base_sin])
            vj = np.column_stack([centers[j, 0] + base_cos, centers[j, 1] + base_sin])
            
            for k in range(6):
                n_vec = normals[k]
                proj_i = np.dot(vi, n_vec)
                proj_j = np.dot(vj, n_vec)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    break
            else:
                return False
    return True


def _separation_violation(centers):
    """
    Compute total separation violation (how much pairs overlap).
    Returns 0 if all pairs are separated, positive value otherwise.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    total_violation = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = np.column_stack([centers[i, 0] + base_cos, centers[i, 1] + base_sin])
            vj = np.column_stack([centers[j, 0] + base_cos, centers[j, 1] + base_sin])
            
            max_overlap = 0.0
            for k in range(6):
                n_vec = normals[k]
                proj_i = np.dot(vi, n_vec)
                proj_j = np.dot(vj, n_vec)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                # Overlap on this axis: how much the intervals overlap
                overlap = min(max_i, max_j) - max(min_i, min_j)
                if overlap > max_overlap:
                    max_overlap = overlap
            
            if max_overlap > 0:
                total_violation += max_overlap ** 2
    
    return total_violation


def _objective(params):
    """Objective: minimize the containing hexagon side length with penalty for overlaps."""
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    # Large penalty for any overlap
    penalty = 100.0 * violation
    return s + penalty


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # Try multiple initial configurations
    initial_configs = [
        _make_good_initial(),
        # Alternative: more spread out
        np.array([
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
        ]),
    ]
    
    best_centers = None
    best_s = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try each initial config with optimization
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
        
        # Run Nelder-Mead optimization
        x0 = init_centers.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 8000,
                    'maxfev': 80000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True,
                }
            )
        except Exception:
            continue
        
        if result is not None:
            candidate = result.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()
        
        # Try a few random restarts from this config
        n_restarts = 3
        for r in range(n_restarts):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            noise_scale = 0.3 * (1.0 + r * 0.5)
            perturbed = init_centers + rng.normal(0, noise_scale, size=init_centers.shape)
            
            x0_p = perturbed.flatten()
            try:
                res_p = minimize(
                    _objective,
                    x0_p,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'maxfev': 50000,
                        'xatol': 1e-9,
                        'fatol': 1e-11,
                        'adaptive': True,
                    }
                )
            except Exception:
                continue
            
            if res_p is not None:
                candidate = res_p.x.reshape(12, 2)
                if _check_separation(candidate):
                    s_cand = _min_containing_hex_side(candidate)
                    if s_cand < best_s:
                        best_s = s_cand
                        best_centers = candidate.copy()
    
    # If we haven't found a valid solution, use a fallback
    if best_centers is None:
        # Use the second initial config as fallback
        best_centers = initial_configs[1].copy()
        best_s = _min_containing_hex_side(best_centers)
    
    # Final refinement: run one more optimization from best solution
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        x0_final = best_centers.flatten()
        try:
            result_final = minimize(
                _objective,
                x0_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'maxfev': 100000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
        except Exception:
            result_final = None
        
        if result_final is not None:
            candidate = result_final.x.reshape(12, 2)
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
