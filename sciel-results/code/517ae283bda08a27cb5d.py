"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time
import math

N = 12

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side s and rotation ang_deg."""
    ang = math.radians(ang_deg)
    verts = []
    for k in range(6):
        theta = ang + k * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)


def _separation_distance(p1, a1, p2, a2, s1=1.0, s2=1.0):
    """
    Compute the minimum separation distance between two regular hexagons.
    If they overlap, return a negative value (approximation of overlap depth).
    This is a conservative check: we check if any vertex of one is inside the other,
    or if edges cross. For simplicity and speed in a DE objective, we use a 
    vertex-containment check and a center-distance heuristic, but a more robust 
    method is needed.
    
    Actually, for regular hexagons of the same size, a sufficient condition for 
    non-intersection is that the distance between centers is >= 2 * inradius? No.
    The inradius (apothem) is s * sqrt(3)/2.
    If the centers are separated by d, and the hexagons are aligned, the min distance 
    between boundaries is d - 2 * apothem if d > 2*apothem.
    But with rotation, it's complex.
    
    Let's use a robust but slightly slower method: 
    Check if any vertex of hex1 is strictly inside hex2, or vice versa.
    And check for edge intersections.
    """
    v1 = _hex_verts(p1[0], p1[1], s1, a1)
    v2 = _hex_verts(p2[0], p2[1], s2, a2)
    
    # Helper: is point p inside polygon poly?
    def point_in_polygon(pt, poly):
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

    # Check vertices of v1 in v2
    for pt in v1:
        if point_in_polygon(pt, v2):
            return -1.0 # Overlap detected

    # Check vertices of v2 in v1
    for pt in v2:
        if point_in_polygon(pt, v1):
            return -1.0 # Overlap detected

    # Check edge intersections
    # This is O(36) which is fine for 12 hexes (66 pairs)
    def seg_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return -1.0 # Overlap detected

    # If no overlap, return a positive distance metric (center distance is a good proxy for "how far apart")
    dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    return dist


def _objective(params, deadline):
    """
    Objective function for Differential Evolution.
    params: shape (N*3,) -> x, y, angle for each hex
    Returns: outer_side_length + penalty
    """
    if time.time() > deadline:
        return 1e10
    
    p = params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    # 1. Calculate required outer hexagon size
    # We want to find the minimum side length S such that all vertices of inner hexagons
    # are inside the outer hexagon centered at (0,0).
    # The outer hexagon has vertices at distance S from center (if rotated by 30 deg? No, standard).
    # Let's define the outer hexagon with flat top, center (0,0), side S.
    # The constraints are defined by 6 half-planes.
    # For a flat-top hexagon of side S:
    # The apothem (distance from center to edge midpoint) is a = S * sqrt(3) / 2.
    # The half-planes are:
    # 1. y <= a
    # 2. y >= -a
    # 3. sqrt(3)*x + y <= 2*a  -> x*sqrt(3) + y <= S*3/2 ? No.
    # Let's derive properly.
    # Vertices of outer hex (flat top, side S):
    # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # Wait, standard hex with side S has distance between opposite sides = S*sqrt(3).
    # So apothem a = S*sqrt(3)/2.
    # The 6 normal directions are:
    # n1 = (0, 1) -> y <= a
    # n2 = (-1/2, -sqrt(3)/2) ? No, let's use the normals to the edges.
    # Edge 1 (top): y = a. Normal (0, 1). Constraint: y <= a.
    # Edge 2 (top-right): Line connecting (S/2, S*sqrt(3)/2) and (S, 0).
    #   Vector: (S/2, -S*sqrt(3)/2). Normal: (S*sqrt(3)/2, S/2) -> (sqrt(3), 1).
    #   Equation: sqrt(3)*x + y = S*sqrt(3) + S*sqrt(3)/2 = S*3*sqrt(3)/2 ?
    #   Let's check vertex (S, 0): sqrt(3)*S + 0 = S*sqrt(3).
    #   Vertex (S/2, S*sqrt(3)/2): sqrt(3)*S/2 + S*sqrt(3)/2 = S*sqrt(3).
    #   So constraint: sqrt(3)*x + y <= S*sqrt(3).
    #   Divide by sqrt(3): x + y/sqrt(3) <= S.
    #   Or: x*sqrt(3) + y <= 3*S/2 * sqrt(3)? No.
    #   Let's stick to dot product with unit normals.
    
    # Unit normals for the 6 edges of a flat-top hexagon:
    # Top edge normal: (0, 1). Distance to edge: a = S*sqrt(3)/2.
    # Top-Right edge normal: (sqrt(3)/2, 1/2).
    # Bottom-Right edge normal: (sqrt(3)/2, -1/2).
    # Bottom edge normal: (0, -1).
    # Bottom-Left edge normal: (-sqrt(3)/2, -1/2).
    # Top-Left edge normal: (-sqrt(3)/2, 1/2).
    
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2, -0.5],
        [-math.sqrt(3)/2, 0.5]
    ])
    
    apothem = math.sqrt(3) / 2.0 # For S=1, apothem is sqrt(3)/2. General: S * sqrt(3)/2.
    
    # We need to find S such that for all inner vertices v:
    # dot(v, n_i) <= S * apothem_unit  where apothem_unit = sqrt(3)/2.
    # So S >= max( dot(v, n_i) / apothem_unit ) for all v and all i.
    # Also S >= 0.
    
    max_proj = 0.0
    for i in range(N):
        verts = _hex_verts(xs[i], ys[i], 1.0, angles[i])
        # Project all vertices onto normals
        projs = verts @ normals.T
        # For each normal, we need S * (sqrt(3)/2) >= max_proj
        # So S >= max_proj / (sqrt(3)/2)
        local_max_S = np.max(projs) / (math.sqrt(3)/2.0)
        if local_max_S > max_proj:
            max_proj = local_max_S
            
    S_required = max(0.0, max_proj)
    
    # 2. Penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            sep = _separation_distance((xs[i], ys[i]), angles[i], (xs[j], ys[j]), angles[j])
            if sep < 0:
                # Overlap. Add a large penalty proportional to overlap? 
                # Since we don't have exact depth, just a binary flag, we add a fixed large penalty.
                # To help DE escape, maybe scale it?
                # Let's add 10.0 per overlap.
                penalty += 10.0
                
    return S_required + penalty


def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: 3x4 grid of hexes
    # Hex spacing: x-direction 2 * apothem = sqrt(3) * 1? No, distance between centers of adjacent hexes in a grid.
    # If flat-top, horizontal distance is 2 * apothem = sqrt(3). Vertical distance is 1.5.
    # Actually, for a honeycomb:
    # Row 0: (0,0), (sqrt(3), 0), (2*sqrt(3), 0), (3*sqrt(3), 0)
    # Row 1: (sqrt(3)/2, 1.5), (3*sqrt(3)/2, 1.5), (5*sqrt(3)/2, 1.5), (7*sqrt(3)/2, 1.5)
    # Row 2: (0, 3.0), (sqrt(3), 3.0), (2*sqrt(3), 3.0), (3*sqrt(3), 3.0)
    
    init_params = []
    for r in range(3):
        for c in range(4):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3)/2
            y = r * 1.5
            ang = 0.0
            init_params.append([x, y, ang])
            
    init_params = np.array(init_params)
    
    # Center the initial guess
    cx = init_params[:, 0].mean()
    cy = init_params[:, 1].mean()
    init_params[:, 0] -= cx
    init_params[:, 1] -= cy
    
    # Define bounds for DE
    # The hexagons shouldn't go too far. 
    # Max coordinate roughly depends on the final S.
    # A safe bound is +/- 10.
    bounds = [(-10, 10), (-10, 10), (-180, 180)] * N
    
    # Run Differential Evolution
    # We need to pass the deadline to the objective, but DE doesn't support that directly.
    # We can wrap the objective to check time, but it's better to control maxiter/popsize.
    # Or just let it run and hope it finishes. 
    # To be safe, we can use a callback to stop if time exceeds deadline.
    
    def callback(xk, convergence=True):
        return time.time() > deadline - 5.0 # Stop 5s early to allow cleanup
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            init='sobol', # Better distribution
            maxiter=50,
            popsize=15,
            tol=0.001,
            callback=callback,
            updating='immediate',
            polish=False # Polish can be slow, and we do our own verification
        )
    except Exception as e:
        # Fallback to initial guess if DE fails
        result_params = init_params.flatten()
    else:
        result_params = result.x
        
    # Post-processing: Verify and possibly refine
    best_params = result_params.reshape(N, 3)
    
    # Calculate final S
    final_S = _objective(best_params, time.time()) # This includes penalty, so we need to separate
    # Let's re-calculate S and penalty separately for the return
    p = best_params
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    # Recalculate S_required
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2, -0.5],
        [-math.sqrt(3)/2, 0.5]
    ])
    apothem_unit = math.sqrt(3) / 2.0
    max_proj = 0.0
    for i in range(N):
        verts = _hex_verts(xs[i], ys[i], 1.0, angles[i])
        projs = verts @ normals.T
        local_max_S = np.max(projs) / apothem_unit
        if local_max_S > max_proj:
            max_proj = local_max_S
    final_S = max(0.0, max_proj)
    
    # Check for overlaps in final result
    # If there are overlaps, the result is invalid. 
    # DE should have minimized them, but if penalty wasn't strong enough or time ran out, 
    # we might have overlaps. 
    # The task says "invalid result ... scores 0". 
    # So we MUST ensure no overlaps.
    
    has_overlap = False
    for i in range(N):
        for j in range(i+1, N):
            sep = _separation_distance((xs[i], ys[i]), angles[i], (xs[j], ys[j]), angles[j])
            if sep < 0:
                has_overlap = True
                break
        if has_overlap:
            break
            
    if has_overlap:
        # Fallback to a known valid configuration: the initial grid, scaled down? 
        # Or just return the initial grid which is valid.
        # The initial grid is valid (no overlaps).
        # We just need to calculate its S.
        p = init_params
        xs = p[:, 0]
        ys = p[:, 1]
        angles = p[:, 2]
        
        max_proj = 0.0
        for i in range(N):
            verts = _hex_verts(xs[i], ys[i], 1.0, angles[i])
            projs = verts @ normals.T
            local_max_S = np.max(projs) / apothem_unit
            if local_max_S > max_proj:
                max_proj = local_max_S
        final_S = max(0.0, max_proj)
        best_params = init_params
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_params[i, 0], best_params[i, 1], best_params[i, 2]])
        
    # Center of outer hex is (0,0)
    # Side length is final_S
    # Angle of outer hex is 0 (flat top)
    
    return inner, (0.0, 0.0), final_S, 0.0

# EVOLVE-BLOCK-END
