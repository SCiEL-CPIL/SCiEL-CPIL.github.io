"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side s and rotation ang_deg."""
    ang = math.radians(ang_deg)
    verts = np.array([
        [cx + s * math.cos(ang + k * math.pi / 3.0),
         cy + s * math.sin(ang + k * math.pi / 3.0)]
        for k in range(6)
    ])
    return verts


def _point_in_polygon(pt, poly):
    """Check if point pt is inside polygon poly using ray casting."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def orient(a, b, c):
        val = (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False


def _on_segment(a, b, c):
    """Check if point b lies on segment a-c."""
    if (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
        min(a[1], c[1]) <= b[1] <= max(a[1], c[1])):
        return True
    return False


def _hexagons_overlap(v1, v2, tol=1e-9):
    """
    Check if two hexagons (given as vertex arrays) overlap.
    Returns True if they overlap, False otherwise.
    """
    # Check if any vertex of v1 is strictly inside v2
    for pt in v1:
        if _point_in_polygon(pt, v2):
            return True
    
    # Check if any vertex of v2 is strictly inside v1
    for pt in v2:
        if _point_in_polygon(pt, v1):
            return True
    
    # Check for edge intersections
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i + 1) % 6], v2[j], v2[(j + 1) % 6]):
                return True
    
    return False


def _min_enclosing_hex_side(centers, angles, side=1.0):
    """
    Compute the minimum side length S of a regular hexagon (centered at origin, 
    with one of the standard orientations) that contains all given hexagons.
    """
    # Unit normals for the 6 edges of a flat-top hexagon
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3) / 2, 0.5],
        [math.sqrt(3) / 2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3) / 2, -0.5],
        [-math.sqrt(3) / 2, 0.5]
    ])
    
    apothem_unit = math.sqrt(3) / 2.0
    max_proj = 0.0
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        verts = _hex_verts(cx, cy, side, ang)
        projs = verts @ normals.T
        local_max_S = np.max(projs) / apothem_unit
        if local_max_S > max_proj:
            max_proj = local_max_S
    
    return max(0.0, max_proj)


def _overlap_penalty(centers, angles, side=1.0, weight=100.0):
    """
    Compute penalty for overlapping hexagons.
    """
    penalty = 0.0
    n = len(centers)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_verts(centers[i][0], centers[i][1], side, angles[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(verts_list[i], verts_list[j]):
                penalty += weight
    
    return penalty


def _objective(params, deadline=None):
    """
    Objective function for optimization.
    params: shape (N*3,) -> x, y, angle for each hex
    Returns: outer_side_length + penalty
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    p = params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    centers = list(zip(xs, ys))
    
    # Calculate required outer hexagon size
    S_required = _min_enclosing_hex_side(centers, angles, side=1.0)
    
    # Penalty for overlaps
    penalty = _overlap_penalty(centers, angles, side=1.0, weight=50.0)
    
    return S_required + penalty


def _make_initial_config():
    """
    Create an initial valid honeycomb lattice configuration for 12 hexagons.
    """
    # Honeycomb lattice vectors for unit hexagons
    # For flat-top hexagons, the lattice vectors are:
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (0, sqrt(3))
    # This gives a 3x4 grid
    v1 = np.array([1.5, math.sqrt(3) / 2])
    v2 = np.array([0.0, math.sqrt(3)])
    
    init_params = []
    for i in range(3):
        for j in range(4):
            pos = i * v1 + j * v2
            init_params.append([pos[0], pos[1], 0.0])
    
    init_params = np.array(init_params)
    
    # Center the configuration
    cx = init_params[:, 0].mean()
    cy = init_params[:, 1].mean()
    init_params[:, 0] -= cx
    init_params[:, 1] -= cy
    
    return init_params


def _try_local_refinement(params, deadline):
    """
    Try to improve the configuration using local optimization.
    """
    if time.time() > deadline:
        return params
    
    best_params = params.copy()
    best_val = _objective(best_params, deadline)
    
    # Try Nelder-Mead from the current best
    try:
        result = minimize(
            _objective,
            best_params,
            args=(deadline,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
        )
        if result.fun < best_val and time.time() < deadline:
            # Verify no overlaps
            p = result.x.reshape(N, 3)
            centers = list(zip(p[:, 0], p[:, 1]))
            angles = p[:, 2]
            penalty = _overlap_penalty(centers, angles, side=1.0, weight=50.0)
            if penalty == 0:
                best_params = result.x
                best_val = result.fun
    except Exception:
        pass
    
    return best_params


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial valid configuration
    init_params = _make_initial_config()
    best_params = init_params.copy()
    
    # Calculate initial objective
    initial_val = _objective(best_params, deadline)
    
    # Run differential evolution
    bounds = [(-8, 8), (-8, 8), (-180, 180)] * N
    
    # Use a seed for reproducibility
    np.random.seed(42)
    
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            args=(deadline,),
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            init='sobol',
            polish=False,
            updating='immediate'
        )
        
        if de_result.fun < initial_val and time.time() < deadline:
            # Verify no overlaps
            p = de_result.x.reshape(N, 3)
            centers = list(zip(p[:, 0], p[:, 1]))
            angles = p[:, 2]
            penalty = _overlap_penalty(centers, angles, side=1.0, weight=50.0)
            if penalty == 0:
                best_params = de_result.x
    except Exception:
        pass
    
    # Try local refinement if time permits
    if time.time() < deadline - 10.0:
        best_params = _try_local_refinement(best_params, deadline)
    
    # Final verification: ensure no overlaps
    p = best_params.reshape(N, 3)
    centers = list(zip(p[:, 0], p[:, 1]))
    angles = p[:, 2]
    penalty = _overlap_penalty(centers, angles, side=1.0, weight=50.0)
    
    if penalty > 0:
        # Fallback to initial valid configuration
        best_params = init_params
        p = best_params.reshape(N, 3)
        centers = list(zip(p[:, 0], p[:, 1]))
        angles = p[:, 2]
    
    # Calculate final S
    final_S = _min_enclosing_hex_side(centers, angles, side=1.0)
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(p[i, 0]), float(p[i, 1]), float(p[i, 2])])
    
    # Center of outer hex is (0,0)
    # Side length is final_S
    # Angle of outer hex is 0 (flat top)
    
    return inner, (0.0, 0.0), float(final_S), 0.0

# EVOLVE-BLOCK-END
