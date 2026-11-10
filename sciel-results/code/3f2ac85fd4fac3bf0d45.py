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
    # Precompute vertices for each unit hexagon (angle 0, side 1)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # All vertices for all hexagons: (n, 6, 2)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i, :, 0] = centers[i, 0] + base_cos
        verts[i, :, 1] = centers[i, 1] + base_sin

    # For each outer edge, compute the max projection of any vertex onto the outward normal
    # Outer hexagon with side s, center (0,0), angle 0:
    # Edge i (from vertex i to vertex i+1) has outward normal direction.
    # The constraint is: for all inner vertices v, dot(v, n_i) <= s * cos(30 deg) = s * sqrt(3)/2
    # Actually, let's derive properly.
    
    # Outer hexagon vertices (side s, center 0, angle 0):
    # v_i = s * (cos(i*60), sin(i*60))
    # Edge from v_i to v_{i+1}. Outward normal n_i.
    # The distance from center to each edge (apothem) is s * cos(30) = s * sqrt(3)/2.
    # The outward normal for edge i points in the direction of the midpoint of the edge's arc,
    # which is at angle (i + 0.5) * 60 degrees.
    
    # For a regular hexagon with side s centered at origin, angle 0:
    # The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330 degrees.
    # The apothem (distance from center to edge) is s * sqrt(3) / 2.
    # So the constraint for each edge j with outward unit normal u_j is:
    #   max over all inner vertices v of dot(v, u_j) <= s * sqrt(3) / 2
    # Therefore: s >= (2 / sqrt(3)) * max_j (max_v dot(v, u_j))
    
    # Outward normals at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each normal, compute max projection over all vertices
    # verts is (n, 6, 2), normals is (6, 2)
    # projections[j, i, k] = dot(verts[i, k], normals[j])
    # We want max over i, k for each j
    
    # Reshape to (n*6, 2)
    verts_flat = verts.reshape(-1, 2)  # (36, 2)
    
    # projections: (6, 36) = normals (6,2) @ verts_flat.T (2, 36)
    projections = normals @ verts_flat.T  # (6, 36)
    
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)  # scalar
    
    s = (2.0 / np.sqrt(3.0)) * max_proj
    
    return s


def _check_separation(centers, tol=1e-6):
    """
    Check if all pairs of unit hexagons are separated using SAT.
    Returns True if all pairs are separated.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # For two axis-aligned hexagons, the SAT axes are the 6 edge normals of a hexagon
    # (since both have the same orientation, we only need 6 axes, not 12)
    # Edge normals at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Vertices of hex i
            vi = np.column_stack([centers[i, 0] + base_cos, centers[i, 1] + base_sin])
            # Vertices of hex j
            vj = np.column_stack([centers[j, 0] + base_cos, centers[j, 1] + base_sin])
            
            for k in range(6):
                n_vec = normals[k]
                proj_i = np.dot(vi, n_vec)
                proj_j = np.dot(vj, n_vec)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                # Check if intervals are separated (gap > tol)
                if max_i < min_j - tol or max_j < min_i - tol:
                    # Found a separating axis, so this pair is separated
                    break
            else:
                # No separating axis found, so they overlap
                return False
    return True


def _objective(params):
    """Objective: minimize the containing hexagon side length."""
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # Initial guess: hexagonal lattice arrangement
    # Use a symmetric arrangement as starting point
    init_centers = np.array([
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
    
    best_centers = init_centers.copy()
    best_s = _min_containing_hex_side(best_centers)
    
    # Check if initial is valid (separated)
    if _check_separation(best_centers):
        pass  # Good start
    
    # Nelder-Mead optimization
    x0 = init_centers.flatten()
    
    result = None
    try:
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 50000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            }
        )
    except Exception:
        pass
    
    if result is not None and result.fun < best_s:
        candidate = result.x.reshape(12, 2)
        if _check_separation(candidate):
            best_centers = candidate
            best_s = result.fun
    
    # Try a few more random restarts if time permits
    rng = np.random.default_rng(42)
    n_restarts = 5
    for r in range(n_restarts):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Perturb the best solution
        noise_scale = 0.5 * (1.0 - r / n_restarts)
        perturbed = best_centers + rng.normal(0, noise_scale, size=best_centers.shape)
        
        x0_p = perturbed.flatten()
        try:
            res_p = minimize(
                _objective,
                x0_p,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'maxfev': 30000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
        except Exception:
            continue
        
        if res_p.fun < best_s:
            candidate = res_p.x.reshape(12, 2)
            if _check_separation(candidate):
                best_centers = candidate
                best_s = res_p.fun
    
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
