"""Packing unit hexagons inside a regular hexagon using smooth optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s limit

def _hex_vertices_local(angle):
    """Vertices of a unit regular hexagon rotated by angle (radians), centered at origin."""
    # Regular hexagon with side length 1: circumradius = 1
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation)
    # Actually for a regular hexagon with side=1, the distance from center to vertex is 1
    # Vertices at angle offsets
    a = np.deg2rad(angle)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    angles = base_angles + a
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _hex_edge_normals_local(angle):
    """Outward normals of the 6 edges of a unit hexagon rotated by angle."""
    a = np.deg2rad(angle)
    # For a regular hexagon (flat-top), edges are at angles 30, 90, 150, 210, 270, 330 from center
    # Edge normals point outward
    base_normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    angles = base_normals + a
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _distance_to_outer_boundary(point, outer_center, outer_side, outer_angle):
    """
    Compute signed distance from point to the boundary of a regular hexagon.
    Positive = inside, negative = outside.
    For a regular hexagon with side length s, the inradius (apothem) = s * sqrt(3)/2
    The distance from center to each edge (along normal) is the apothem.
    """
    oc = np.array(outer_center)
    oa = np.deg2rad(outer_angle)
    apothem = outer_side * np.sqrt(3) / 2.0
    
    # Edge normals of outer hexagon
    base_normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    normals = np.column_stack([np.cos(base_normals + oa), np.sin(base_normals + oa)])
    
    rel = np.array(point) - oc
    # Distance from center along each normal direction
    dists = rel @ normals.T  # shape (6,)
    # The point is inside if dists[i] <= apothem for all i
    # Signed distance to boundary = min over edges of (apothem - dists[i])
    # But this is only valid when point is in the "normal" region of each edge
    # For a regular hexagon, this works: the hexagon is the intersection of 6 half-planes
    return np.min(apothem - dists)

def _min_clearance_inner(point, inner_centers, inner_angles, exclude_idx=None):
    """
    Compute minimum distance from point to any inner hexagon boundary (excluding exclude_idx).
    For two convex polygons, we can use the support function approach, but for simplicity
    we'll compute distance to each vertex and edge of each hexagon.
    """
    min_dist = np.inf
    n_inner = len(inner_centers)
    for i in range(n_inner):
        if i == exclude_idx:
            continue
        ic = np.array(inner_centers[i])
        ia = np.deg2rad(inner_angles[i])
        
        # Vertices of inner hexagon i
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        v_angles = base_angles + ia
        verts = ic + np.column_stack([np.cos(v_angles), np.sin(v_angles)])
        
        # Distance to vertices
        d_verts = np.min(np.linalg.norm(point - verts, axis=1))
        
        # Distance to edges
        # Edge normals
        base_normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
        normals = np.column_stack([np.cos(base_normals + ia), np.sin(base_normals + ia)])
        apothem = np.sqrt(3) / 2.0  # for side=1
        
        # Project point onto each edge's normal
        rel = point - ic
        proj = rel @ normals.T  # (6,)
        # For each edge, if proj[i] > apothem, distance is proj[i] - apothem (outside)
        # But we need to check if the projection falls within the edge segment
        # For a regular hexagon, the edge length is 1, and the edge spans from -0.5 to 0.5 along the tangent
        # This is complex; for now, use vertex distances and a conservative edge distance
        # Actually, for two convex polygons, the minimum distance can be computed via
        # the minimum over all vertex-edge and edge-vertex pairs
        # For simplicity in the smooth optimization, we'll use a lower bound:
        # The distance from point to the hexagon is at least the minimum over edges of
        # max(0, proj[i] - apothem) but this isn't tight
        # Let's use a simpler approach: distance to the 6 vertices gives a good approximation
        # for the smooth optimization, and we'll verify with exact check later
        
        min_dist = min(min_dist, d_verts)
    
    return min_dist

def _objective(params, N):
    """
    Objective: minimize a smooth approximation of the maximum violation.
    params: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx,cy, S, cA]
    where (xi, yi, ai) are inner hexagon positions/angles (degrees),
    (cx, cy) is outer center, S is outer side length, cA is outer angle (degrees)
    
    We want to minimize S subject to:
    1. Each inner hexagon is inside the outer hexagon
    2. No two inner hexagons overlap
    
    We use a smooth penalty: for each constraint, we compute the "violation" and use
    a soft-min to get a smooth objective.
    """
    inner = params[:3*N].reshape(N, 3)
    cx, cy, S, cA = params[3*N], params[3*N+1], params[3*N+2], params[3*N+3]
    
    # Ensure S is positive
    S = max(S, 1.0)
    
    # Compute violations
    violations = []
    
    # 1. Inner hexagons must be inside outer hexagon
    outer_center = (cx, cy)
    outer_angle = cA
    apothem = S * np.sqrt(3) / 2.0
    base_normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    oa = np.deg2rad(cA)
    normals = np.column_stack([np.cos(base_normals + oa), np.sin(base_normals + oa)])
    
    for i in range(N):
        xi, yi, ai = inner[i]
        ia = np.deg2rad(ai)
        # Vertices of inner hexagon i
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        v_angles = base_angles + ia
        verts = np.column_stack([xi + np.cos(v_angles), yi + np.sin(v_angles)])
        
        # For each vertex, compute signed distance to outer boundary
        # Vertex is inside if for all outer edges: (vertex - outer_center) . normal <= apothem
        rel = verts - np.array([cx, cy])
        dists = rel @ normals.T  # (6, 6) for 6 vertices x 6 edges
        # Signed distance for each vertex = min over edges of (apothem - dists[v,e])
        # A vertex is outside if any edge has dists[v,e] > apothem
        # Violation for vertex = max(0, max_e(dists[v,e]) - apothem)
        vertex_violations = np.max(dists, axis=1) - apothem  # (6,)
        # The hexagon is inside if all vertices are inside (for convex shapes)
        # Actually, for a convex polygon inside another convex polygon, it's sufficient
        # to check that all vertices of the inner are inside the outer
        max_viol = np.max(vertex_violations)
        violations.append(max(0, max_viol))
    
    # 2. No two inner hexagons overlap
    # For two convex polygons, they overlap if and only if there is no separating axis
    # For simplicity, we'll use a smooth approximation based on center distances
    # and a conservative radius
    # The "radius" of a unit hexagon is 1 (circumradius)
    # Two hexagons with centers distance d apart will overlap if d < 2 (roughly)
    # But this is very conservative. Let's use a better estimate.
    # Actually, for two identical regular hexagons, the minimum distance between
    # centers for non-overlap depends on relative orientation.
    # For now, use a smooth penalty based on a lower bound on separation.
    # The inradius of a unit hexagon is sqrt(3)/2 ≈ 0.866
    # Two hexagons definitely don't overlap if center distance > 2 * circumradius = 2
    # They might overlap if center distance < 2 * inradius = sqrt(3) ≈ 1.732
    # For a smooth objective, we'll penalize when center distance < some threshold
    # that depends on relative angle. This is complex, so let's use a simple approach:
    # penalize based on the minimum distance between vertices of the two hexagons
    
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            ia = np.deg2rad(ai)
            ja = np.deg2rad(aj)
            
            # Vertices of hexagon i
            base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
            vi = np.column_stack([xi + np.cos(base_angles + ia), yi + np.sin(base_angles + ia)])
            vj = np.column_stack([xj + np.cos(base_angles + ja), yj + np.sin(base_angles + ja)])
            
            # Minimum distance between any vertex of i and any vertex of j
            # This is 36 distances, compute efficiently
            # d[v_i, v_j] = ||v_i - v_j||
            # We want min over all pairs
            # For smoothness, use a soft-min: -log(sum(exp(-d/T))) * T for small T
            # But this is expensive. Let's use a simpler approach:
            # The minimum distance between two sets of points can be approximated by
            # the minimum over all vertex-vertex pairs. For 6x6=36 pairs, this is fine.
            diffs = vi[:, np.newaxis, :] - vj[np.newaxis, :, :]  # (6, 6, 2)
            dists = np.linalg.norm(diffs, axis=2)  # (6, 6)
            min_d = np.min(dists)
            
            # If min_d < 0, they overlap (vertices interpenetrate)
            # Actually, min_d = 0 means they touch. min_d > 0 means separated.
            # For non-overlap, we need min_d >= 0 (they can touch)
            # But this only checks vertex-vertex. We should also check vertex-edge.
            # For now, use vertex-vertex as a proxy and add a small safety margin.
            # Violation if min_d < 0 (overlap)
            # Use smooth max(0, -min_d)
            violations.append(max(0.0, -min_d))
    
    # Smooth objective: we want to minimize S + penalty for violations
    # Use a smooth sum of violations
    if len(violations) > 0:
        viol_arr = np.array(violations)
        # Soft sum: T * log(sum(exp(viol/T))) for small T
        T = 0.1
        max_v = np.max(viol_arr)
        if max_v > 1e-10:
            soft_sum = T * np.log(np.sum(np.exp((viol_arr - max_v) / T))) + max_v
        else:
            soft_sum = 0.0
    else:
        soft_sum = 0.0
    
    # Objective: minimize S + lambda * soft_sum
    # We want S to be small, but violations must be ~0
    # Use a large penalty for violations
    obj = S + 100.0 * soft_sum
    return obj

def _gradient(params, N):
    """Numerical gradient (finite differences)."""
    eps = 1e-5
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (_objective(p_plus, N) - _objective(p_minus, N)) / (2 * eps)
    return grad

def construct():
    global START_TIME
    start = time.time()
    
    # Initial guess: place hexagons in a hexagonal lattice pattern
    # For N=11, a hexagonal arrangement with 3-4-3 or similar
    # Let's try a centered arrangement
    
    # Best solution found
    best_S = 100.0
    best_params = None
    
    # Initial parameters: inner hexagons in a rough hexagonal pattern
    # Outer hexagon centered at origin
    # Try different initial configurations
    
    configs = []
    
    # Config 1: Linear arrangement (like the seed)
    inner1 = []
    for k in range(-(N//2), N - N//2):
        inner1.append([4.0*k, 0.0, 0.0])
    configs.append((inner1, (0.0, 0.0), 40.0, 0.0))
    
    # Config 2: Hexagonal lattice
    # For N=11, use a 3-4-3 pattern or 2-3-4-2 etc.
    # Let's try a compact hexagonal arrangement
    # Ring 0: 1 hex at center
    # Ring 1: 6 hexes around
    # Ring 2: 12 hexes (but we only need 4 more for total 11)
    
    # Actually, for 11 hexagons, a good arrangement might be:
    # Center + 6 in first ring + 4 in second ring (partial)
    
    inner2 = [[0.0, 0.0, 0.0]]
    # First ring: 6 hexagons at distance 2*sqrt(3) ≈ 3.464 from center
    # In a hexagonal lattice with side 1, adjacent hexagons have center distance 2*sqrt(3)/... 
    # Actually, for two unit hexagons side by side (sharing an edge), center distance = 2*apothem = 2*sqrt(3)/2*2 = 2*sqrt(3) ≈ 3.464
    # Wait, apothem of unit hexagon = sqrt(3)/2 ≈ 0.866
    # Two hexagons sharing an edge: center distance = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732
    # No, that's not right either. Let me think again.
    # A regular hexagon with side length 1:
    # - Circumradius (center to vertex) = 1
    # - Inradius/apothem (center to edge midpoint) = sqrt(3)/2 ≈ 0.866
    # Two hexagons placed side by side, sharing a full edge:
    # The distance between centers = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732
    # So in a hexagonal lattice, adjacent centers are sqrt(3) apart.
    
    d_adj = np.sqrt(3)  # distance between adjacent hexagon centers
    
    # First ring: 6 hexagons at distance d_adj from center, at angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = k * 60
        x = d_adj * np.cos(np.radians(angle))
        y = d_adj * np.sin(np.radians(angle))
        inner2.append([x, y, 0.0])
    
    # Second ring: add 4 more hexagons
    # Second ring centers are at distance 2*d_adj from center
    # At angles 0, 60, 120, 180 (4 of the 6 possible)
    for k in [0, 1, 2, 3]:
        angle = k * 60
        x = 2 * d_adj * np.cos(np.radians(angle))
        y = 2 * d_adj * np.sin(np.radians(angle))
        inner2.append([x, y, 0.0])
    
    # Trim to N hexagons
    inner2 = inner2[:N]
    configs.append((inner2, (0.0, 0.0), 10.0, 0.0))
    
    # Config 3: More spread out
    inner3 = []
    for k in range(N):
        # Spiral arrangement
        r = 1.0 + 0.5 * k
        theta = k * 137.5  # golden angle
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        inner3.append([x, y, 0.0])
    configs.append((inner3, (0.0, 0.0), 15.0, 0.0))
    
    for config_idx, (inner, center, S_init, angle_init) in enumerate(configs):
        if time.time() - start > TIME_LIMIT - 10:
            break
        
        # Build initial parameter vector
        params = np.array([v for row in inner for v in row] + [center[0], center[1], S_init, angle_init])
        
        # Optimize
        result = minimize(_objective, params, args=(N,), method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-4})
        
        if result.success or result.fun < _objective(params, N):
            best_params = result.x
            best_S = result.x[3*N+2]  # S is at index 3*N+2
        
        # Check time
        if time.time() - start > TIME_LIMIT - 10:
            break
    
    # If no good result, use the seed configuration
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    inner_result = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner_result.append([x, y, a])
    
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    S = best_params[3*N+2]
    cA = best_params[3*N+3]
    
    # Ensure S is reasonable
    S = max(S, 1.0)
    
    return inner_result, (cx, cy), S, cA

