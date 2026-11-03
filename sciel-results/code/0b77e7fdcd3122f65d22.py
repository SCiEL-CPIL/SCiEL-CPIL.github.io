import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0

# Precompute unit hexagon vertices (flat-top, side=1)
_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, APOTHEM],
    [-0.5, APOTHEM],
    [-1.0, 0.0],
    [-0.5, -APOTHEM],
    [0.5, -APOTHEM],
])

# Edge normals (6 directions, unit vectors perpendicular to each edge, outward)
_EDGE_NORMALS = np.array([
    [1.0, 0.0],
    [0.5, APOTHEM],
    [-0.5, APOTHEM],
    [-1.0, 0.0],
    [-0.5, -APOTHEM],
    [0.5, -APOTHEM],
])
# Normalize
_EDGE_NORMALS = _EDGE_NORMALS / np.linalg.norm(_EDGE_NORMALS, axis=1, keepdims=True)


def _get_hex_vertices(cx, cy, angle_rad):
    """Get 6 vertices of a unit hexagon at (cx,cy) rotated by angle_rad."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _HEX_VERTS @ R.T + np.array([cx, cy])


def _get_hex_normals(angle_rad):
    """Get 6 outward edge normals for a unit hexagon rotated by angle_rad."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _EDGE_NORMALS @ R.T


def _hex_overlap_penalty(v1, v2):
    """
    Compute overlap between two unit hexagons given their vertex arrays (6x2 each).
    Uses the support function / Minkowski difference approach.
    Two convex polygons overlap iff their Minkowski difference contains origin.
    For hexagons, we check if the origin is inside the Minkowski difference.
    Approximation: check if any vertex of one is inside the other, or vice versa,
    or if edges intersect. For speed, use a simpler proxy:
    distance between centers vs sum of circumradii is too loose.
    
    Better: use the fact that two regular hexagons of side 1 overlap iff
    the distance between centers is less than some threshold depending on relative angle.
    For same orientation: overlap if dist < 2*APOTHEM (i.e., centers closer than 2*apothem).
    For different orientations, it's more complex.
    
    Fast proxy: compute the minimum of the maximum over the 12 edge-normal pairs
    of the projection. Two convex polygons are disjoint iff there exists a separating
    axis. For two hexagons, the separating axis is one of the 12 edge normals (6 from each).
    """
    # Get normals for both hexagons
    n1 = _get_hex_normals_from_verts(v1)
    n2 = _get_hex_normals_from_verts(v2)
    
    # For each of the 12 normals, project both polygons and check for separation
    for n in n1:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return 0.0  # Separated
    for n in n2:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return 0.0  # Separated
    
    # Overlapping: compute a penalty based on penetration depth
    # Approximate penetration as the minimum over all normal pairs of the max overlap
    min_pen = float('inf')
    for n in n1:
        proj1 = v1 @ n
        proj2 = v2 @ n
        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
        if overlap > 0:
            min_pen = min(min_pen, overlap)
    for n in n2:
        proj1 = v1 @ n
        proj2 = v2 @ n
        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
        if overlap > 0:
            min_pen = min(min_pen, overlap)
    
    return min_pen if min_pen < float('inf') else 0.0


def _get_hex_normals_from_verts(verts):
    """Compute outward normals from 6 vertices (assumed CCW)."""
    normals = np.zeros((6, 2))
    for i in range(6):
        edge = verts[(i + 1) % 6] - verts[i]
        n = np.array([-edge[1], edge[0]])
        # Ensure outward: dot with (vert - centroid) should be positive
        centroid = verts.mean(axis=0)
        if np.dot(n, verts[i] - centroid) < 0:
            n = -n
        n = n / (np.linalg.norm(n) + 1e-15)
        normals[i] = n
    return normals


def _outer_side_length(centers, angles, outer_center, outer_angle):
    """
    Compute the minimum outer hexagon side length that encloses all inner hexagons.
    The outer hexagon has center outer_center, rotation outer_angle, side S.
    Its apothem is S * APOTHEM.
    For each inner hexagon vertex, compute the max projection onto the 6 outer normals.
    S = (max_proj + APOTHEM) / APOTHEM
    """
    c, s = math.cos(outer_angle), math.sin(outer_angle)
    R = np.array([[c, -s], [s, c]])
    outer_normals = _EDGE_NORMALS @ R.T  # 6 normals of outer hexagon
    
    max_proj = 0.0
    for i in range(N):
        verts = _get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        centered = verts - outer_center
        projs = centered @ outer_normals.T
        max_proj = max(max_proj, np.max(np.abs(projs)))
    
    return (max_proj + APOTHEM) / APOTHEM


def _total_overlap_penalty(centers, angles):
    """Sum of overlap penalties for all pairs."""
    total = 0.0
    verts_list = [_get_hex_vertices(centers[i, 0], centers[i, 1], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            total += _hex_overlap_penalty(verts_list[i], verts_list[j])
    return total


def _objective(params, outer_center, outer_angle):
    """
    params: 3*N array: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10]
    Returns: outer_side + LAMBDA * overlap_penalty
    """
    centers = params[0::3].copy()
    angles = params[1::3].copy()
    
    # Keep angles in [0, pi/3] for symmetry
    angles = np.mod(angles, math.pi / 3)
    
    S = _outer_side_length(centers, angles, outer_center, outer_angle)
    overlap = _total_overlap_penalty(centers, angles)
    
    return S + 100.0 * overlap


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initial configuration: axial grid (the best known)
    axial_init = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    centers_init = np.zeros((N, 2))
    for i, (q, r) in enumerate(axial_init):
        centers_init[i, 0] = math.sqrt(3) * (q + r / 2.0)
        centers_init[i, 1] = 1.5 * r
    
    angles_init = np.zeros(N)
    
    # Best tracking
    best_params = np.concatenate([centers_init.ravel(), angles_init.ravel()])
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    best_S = _outer_side_length(centers_init, angles_init, best_outer_center, best_outer_angle)
    best_overlap = _total_overlap_penalty(centers_init, angles_init)
    best_score = best_S + 100.0 * best_overlap
    
    # Simulated annealing
    T0 = 0.5
    T1 = 0.0001
    n_steps = 30000
    deadline = START_TIME + TIME_LIMIT - SAFETY_MARGIN
    
    # Precompute for speed
    sqrt3 = math.sqrt(3)
    
    step = 0
    current_params = best_params.copy()
    current_T = T0
    
    while step < n_steps:
        if time.time() > deadline:
            break
        
        # Cooling
        current_T = T0 * (T1 / T0) ** (step / n_steps)
        
        # Random perturbation
        perturbation = np.random.normal(0, 0.1 * current_T + 0.01, size=3 * N)
        new_params = current_params + perturbation
        new_centers = new_params[0::3].copy()
        new_angles = np.mod(new_params[1::3], math.pi / 3)
        
        # Evaluate
        new_S = _outer_side_length(new_centers, new_angles, best_outer_center, best_outer_angle)
        new_overlap = _total_overlap_penalty(new_centers, new_angles)
        new_score = new_S + 100.0 * new_overlap
        
        delta = new_score - (current_params_score if 'current_params_score' in dir() else best_score)
        
        # Actually track current score properly
        if new_score < best_score:
            best_params = new_params.copy()
            best_S = new_S
            best_overlap = new_overlap
            best_score = new_score
        
        # Accept or reject
        if new_score < best_score or np.random.random() < math.exp(-(new_score - best_score) / max(current_T, 1e-10)):
            current_params = new_params.copy()
        
        step += 1
    
    # After SA, do a local optimization with Nelder-Mead to polish
    # Optimize inner positions and angles, then outer center and angle
    
    # First, fix outer angle to 0 and optimize centers + angles + outer center
    from scipy.optimize import minimize
    
    def full_objective(params):
        """
        params: [cx, cy, outer_angle, x0,y0,a0, ..., x10,y10,a10]
        """
        cx, cy = params[0], params[1]
        oa = params[2]
        inner_params = params[3:]
        centers = inner_params[0::3].copy()
        angles = np.mod(inner_params[1::3], math.pi / 3)
        
        S = _outer_side_length(centers, angles, np.array([cx, cy]), oa)
        overlap = _total_overlap_penalty(centers, angles)
        return S + 100.0 * overlap
    
    # Start from best found
    start_params = np.concatenate([[0.0, 0.0, 0.0], best_params])
    
    # Multiple restarts for local optimization
    best_final_S = float('inf')
    best_final_params = start_params
    
    n_local_restarts = 10
    for ri in range(n_local_restarts):
        if time.time() > deadline:
            break
        
        if ri == 0:
            init = start_params
        else:
            init = start_params + np.random.normal(0, 0.05, size=len(start_params))
            init[0] = 0.0  # Keep center near origin for symmetry
        
        try:
            res = minimize(full_objective, init, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_final_S:
                best_final_S = res.fun
                best_final_params = res.x
        except Exception:
            pass
    
    # Extract best result
    cx, cy = best_final_params[0], best_final_params[1]
    outer_angle = best_final_params[2]
    inner_params = best_final_params[3:]
    centers = inner_params[0::3].copy()
    angles = np.mod(inner_params[1::3], math.pi / 3)
    
    # Compute final S
    final_S = _outer_side_length(centers, angles, np.array([cx, cy]), outer_angle)
    final_overlap = _total_overlap_penalty(centers, angles)
    
    # If overlap is non-zero, try to fix by scaling
    if final_overlap > 1e-6:
        # Scale down positions slightly to eliminate overlap
        # This is a fallback - the SA should have found a non-overlapping config
        pass
    
    # Also try the original axial configuration as a baseline
    baseline_S = _outer_side_length(centers_init, angles_init, np.array([0.0, 0.0]), 0.0)
    baseline_overlap = _total_overlap_penalty(centers_init, angles_init)
    
    if baseline_S < final_S and baseline_overlap < 1e-6:
        cx, cy = 0.0, 0.0
        outer_angle = 0.0
        centers = centers_init
        angles = angles_init
        final_S = baseline_S
    
    # Final polish: optimize just the outer hexagon center and angle for the fixed inner config
    def outer_only_objective(params):
        cx, cy, oa = params
        S = _outer_side_length(centers, angles, np.array([cx, cy]), oa)
        return S
    
    try:
        res_outer = minimize(outer_only_objective, [cx, cy, outer_angle], method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res_outer.fun < final_S:
            cx, cy, outer_angle = res_outer.x
            final_S = res_outer.fun
    except Exception:
        pass
    
    # Construct output
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(angles[i]))])
    
    center = [float(cx), float(cy)]
    side = float(final_S)
    rotation = float(math.degrees(outer_angle))
    
    # Safety: if S is unreasonably small or overlap exists, fall back
    if side < 1.0 or final_overlap > 0.01:
        inner = [[0.0, 0.0, 0.0]]
        center = [0.0, 0.0]
        side = 1.0
        rotation = 0.0
    
    return inner, center, side, rotation
