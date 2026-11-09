"""Parametric hexagonal lattice packing of 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12
DEADLINE = 120.0  # wall-clock budget with safety margin

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    r = side
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3.0
        verts.append([cx + r * np.cos(theta), cy + r * np.sin(theta)])
    return np.array(verts)

def _hex_face_normals(angle_deg):
    """Outward face normals (unit) for a regular hexagon at given rotation."""
    a = np.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + (2 * k + 1) * np.pi / 6.0
        normals.append([np.cos(theta), np.sin(theta)])
    return np.array(normals)

def _min_enclosing_hexagon(inner_positions, inner_angles, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon (with given rotation)
    that encloses all given inner hexagons.
    
    Uses the property that a regular hexagon can be described as the intersection
    of 6 half-planes. For a hexagon centered at c with side s and rotation a,
    a point p is inside iff for each face normal n_i: n_i . (p - c) <= s * sqrt(3)/2.
    
    So the minimum s for a given center c and rotation a is:
        s = max over all vertices p and all face normals n_i of [ n_i . (p - c) ] / (sqrt(3)/2)
    
    But we also need to optimize the center c. For a fixed rotation, the optimal
    center is the midpoint of the minimal enclosing region in each normal direction.
    """
    sqrt3_over_2 = np.sqrt(3.0) / 2.0
    normals = _hex_face_normals(outer_angle_deg)  # shape (6, 2)
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in inner_positions:
        v = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # shape (M, 2)
    
    # For each face normal direction, compute the min and max projection
    # The hexagon must contain all points, so for each normal n_i:
    #   min_p n_i . p <= n_i . c and n_i . c + s*sqrt3/2 >= max_p n_i . p
    # So: n_i . c >= max_p n_i . p - s*sqrt3/2  AND  n_i . c <= min_p n_i . p + s*sqrt3/2
    # This is a feasibility problem in (c_x, c_y, s).
    
    # For a given s, we need: for each i,
    #   max_p n_i . p - s*sqrt3/2 <= n_i . c <= min_p n_i . p + s*sqrt3/2
    # And also for the opposite direction (-n_i is not a face normal of hexagon, 
    # but the hexagon has 6 faces with normals at angles a + (2k+1)*pi/6).
    # Actually the 6 normals already cover all 6 directions.
    
    # The condition is: for all i, n_i . c must be in [max_proj_i - s*sqrt3/2, min_proj_i + s*sqrt3/2]
    # where max_proj_i = max_p n_i . p, min_proj_i = min_p n_i . p
    # Note: for a hexagon, the opposite face has normal -n_{i+3} (since normals are 60 deg apart, 
    # opposite is 180 deg = 3 steps). So n_{i+3} = -n_i.
    # Thus min_proj_i = -max_proj_{i+3} and max_proj_i = -min_proj_{i+3}.
    
    # Let's compute projections
    projections = all_verts @ normals.T  # shape (M, 6)
    max_proj = projections.max(axis=0)  # shape (6,)
    min_proj = projections.min(axis=0)  # shape (6,)
    
    # For feasibility: for each i, we need n_i . c in [max_proj[i] - s*sqrt3/2, min_proj[i] + s*sqrt3/2]
    # Note: min_proj[i] + s*sqrt3/2 >= max_proj[i] - s*sqrt3/2  =>  s*sqrt3 >= max_proj[i] - min_proj[i]
    # So s >= (max_proj[i] - min_proj[i]) / sqrt3 for all i.
    # But that's just the width in that direction. The center also needs to be chosen.
    
    # Actually, let me think differently. A regular hexagon with center c, side s, rotation a
    # is the set of points p such that for each of the 6 face normals n_i:
    #   n_i . (p - c) <= s * sqrt3 / 2
    # Equivalently: n_i . p <= n_i . c + s * sqrt3 / 2
    # For all points p to be inside: for all i, max_p(n_i . p) <= n_i . c + s * sqrt3 / 2
    # => n_i . c >= max_proj[i] - s * sqrt3 / 2
    # Also, since the hexagon is symmetric, the opposite face gives:
    #   (-n_i) . (p - c) <= s * sqrt3 / 2  =>  -n_i . p <= -n_i . c + s * sqrt3 / 2
    #   => n_i . p >= n_i . c - s * sqrt3 / 2
    #   => min_p(n_i . p) >= n_i . c - s * sqrt3 / 2
    #   => n_i . c <= min_proj[i] + s * sqrt3 / 2
    #
    # So we need: for all i:  max_proj[i] - s*sqrt3/2 <= n_i . c <= min_proj[i] + s*sqrt3/2
    #
    # This is a linear feasibility problem in (c_x, c_y, s). We want to minimize s.
    # We can use binary search on s and check feasibility via linear programming, 
    # or we can note that the optimal s is the maximum over i of (width_i + something).
    #
    # Actually, for a fixed rotation, the problem of finding the minimum enclosing regular hexagon
    # can be solved by noting that the optimal center c is determined by the tightest constraints.
    # Let's use a different approach: parameterize by s and check if there exists c.
    
    # For a given s, the constraints are:
    #   n_i . c >= max_proj[i] - s*sqrt3/2   for i = 0..5
    #   n_i . c <= min_proj[i] + s*sqrt3/2    for i = 0..5
    # This is a system of 12 linear inequalities in 2 unknowns (c_x, c_y).
    # We can check feasibility using linear programming (minimize 0 subject to constraints).
    
    # But doing LP for each s in a binary search is slow. Let's think of a direct formula.
    # 
    # Alternative: The minimum enclosing regular hexagon (with fixed rotation) can be found by
    # computing for each of the 6 normal directions the "required extent" and then finding the 
    # smallest s such that the intersection of the 6 strips is non-empty.
    #
    # Let me use scipy.optimize to solve this as a convex optimization problem:
    # minimize s
    # subject to: n_i . c >= max_proj[i] - s*sqrt3/2  for all i
    #              n_i . c <= min_proj[i] + s*sqrt3/2   for all i
    #              s >= 0
    #
    # This is a linear program! Variables: c_x, c_y, s (3 variables).
    # Constraints: 12 linear inequalities + s >= 0.
    
    # Let's set up the LP.
    # Minimize: [0, 0, 1] . [c_x, c_y, s]
    # Constraints:
    #   n_i . c - s*sqrt3/2 >= max_proj[i]  =>  n_i[0]*c_x + n_i[1]*c_y - sqrt3/2*s >= max_proj[i]
    #   -n_i . c - s*sqrt3/2 >= -min_proj[i]  =>  -n_i[0]*c_x - n_i[1]*c_y - sqrt3/2*s >= -min_proj[i]
    #   s >= 0
    
    # Using scipy.optimize.linprog: minimize c^T x subject to A_ub x <= b_ub, A_eq x = b_eq
    # Convert >= to <=: multiply by -1.
    
    A_ub = []
    b_ub = []
    for i in range(6):
        # n_i . c - sqrt3/2 * s >= max_proj[i]  =>  -n_i[0]*c_x - n_i[1]*c_y + sqrt3/2*s <= -max_proj[i]
        A_ub.append([-normals[i, 0], -normals[i, 1], sqrt3_over_2])
        b_ub.append(-max_proj[i])
        # -n_i . c - sqrt3/2 * s >= -min_proj[i]  =>  n_i[0]*c_x + n_i[1]*c_y + sqrt3/2*s <= min_proj[i]
        A_ub.append([normals[i, 0], normals[i, 1], sqrt3_over_2])
        b_ub.append(min_proj[i])
    # s >= 0  =>  -s <= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    c_obj = np.array([0.0, 0.0, 1.0])
    
    from scipy.optimize import linprog
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if res.success:
        return res.fun
    else:
        # Fallback: return a large value
        return 1e10

def _lattice_positions(scale, row_shift, global_rotation):
    """
    Generate 12 hexagon positions on a parametric hexagonal lattice.
    
    The lattice has rows of hexagons. For a hexagonal close-packed arrangement:
    - Row 0: 3 hexagons
    - Row 1: 4 hexagons  
    - Row 2: 3 hexagons
    - Row 3: 2 hexagons
    Total: 12
    
    Actually, let's use a more standard hexagonal packing:
    Row 0: 2 hexagons
    Row 1: 3 hexagons
    Row 2: 4 hexagons
    Row 3: 3 hexagons
    Total: 12
    
    Or:
    Row 0: 3
    Row 1: 4
    Row 2: 3
    Row 3: 2
    Total: 12
    
    Let's use a hexagonal grid with variable spacing.
    """
    positions = []
    
    # Define row structure: number of hexagons per row
    # Try a few topologies and pick the best one
    # Topology 1: 3-4-3-2
    row_counts = [3, 4, 3, 2]
    
    # Vertical spacing between rows
    vert_spacing = scale * np.sqrt(3) / 2.0
    
    for row_idx, count in enumerate(row_counts):
        # Horizontal offset for this row
        h_offset = row_shift * (row_idx % 2)  # staggered rows
        
        for col_idx in range(count):
            # Center the row horizontally
            x = (col_idx - (count - 1) / 2.0) * scale + h_offset
            y = (row_idx - (len(row_counts) - 1) / 2.0) * vert_spacing
            positions.append([x, y, global_rotation])
    
    return positions

def _objective(params):
    """
    Objective function: minimize the outer hexagon side length.
    params: [scale, row_shift, global_rotation, outer_rotation]
    """
    scale, row_shift, global_rotation, outer_rotation = params
    
    if scale <= 0:
        return 1e10
    
    positions = _lattice_positions(scale, row_shift, global_rotation)
    
    if len(positions) != N:
        return 1e10
    
    # Check for overlaps (approximate: center-to-center distance must be >= 2 for unit hexagons
    # with same orientation, but with rotation it's more complex)
    # For now, just compute the enclosing hexagon side length
    
    side = _min_enclosing_hexagon(positions, [p[2] for p in positions], outer_rotation)
    return side

def construct():
    start_time = time.time()
    
    # Initial guess
    best_params = [2.0, 0.0, 0.0, 0.0]
    best_side = _objective(best_params)
    
    # Use differential evolution for global optimization
    bounds = [(1.5, 4.0), (-1.0, 1.0), (-30.0, 30.0), (-30.0, 30.0)]
    
    elapsed = time.time() - start_time
    if elapsed < 60:
        try:
            result = differential_evolution(
                _objective, bounds, 
                maxiter=100, popsize=15,
                tol=1e-6, seed=42,
                polish=True
            )
            if result.success and result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < 100:
        try:
            result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Final polish
    elapsed = time.time() - start_time
    if elapsed < 115:
        try:
            result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except Exception:
            pass
    
    scale, row_shift, global_rotation, outer_rotation = best_params
    positions = _lattice_positions(scale, row_shift, global_rotation)
    
    # Ensure we have exactly N positions
    while len(positions) < N:
        positions.append([0.0, 0.0, global_rotation])
    positions = positions[:N]
    
    # Convert to required format: (x, y, angle_degrees)
    inner = [[p[0], p[1], p[2]] for p in positions]
    
    return inner, (0.0, 0.0), best_side, outer_rotation
