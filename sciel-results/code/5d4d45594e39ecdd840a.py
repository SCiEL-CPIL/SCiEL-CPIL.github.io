import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)
TIME_LIMIT = 145.0 # Safety margin for 150s kill

# Precompute base vertices for a unit hexagon (side 1)
# Vertices at angles 0, 60, ..., 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_vertices(cx, cy, angle_rad):
    """
    Returns (6, 2) vertices for a unit hexagon.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ BASE_VERTS.T).T + np.array([cx, cy])

def compute_outer_side_length(vertices):
    """
    vertices: (N*6, 2) array of all vertices.
    Returns the minimum side length of a regular hexagon centered at origin 
    that contains all vertices.
    The side length L of a regular hexagon centered at origin is determined by 
    the maximum projection of any point onto the 6 normal directions.
    The support function h(u) = max(x . u).
    For a regular hexagon with side s, the distance from center to edge (apothem) is s * sqrt(3)/2.
    So if max projection is P, then P = s * sqrt(3)/2 => s = P * 2 / sqrt(3).
    Wait, let's verify.
    Hexagon vertices at distance s from center.
    Normals are at angles 0, 60, 120, 180, 240, 300.
    Projection of vertex (s, 0) onto (1, 0) is s.
    So max projection P = s.
    Thus side length = P.
    Let's check.
    If hexagon has side s, vertices are at distance s from center.
    The "width" in the direction of a vertex is s.
    The "width" in the direction of an edge normal is s * sqrt(3).
    Wait, the problem says "outer side length".
    A regular hexagon is defined by its side length.
    If we have a set of points, the smallest regular hexagon containing them 
    is determined by the maximum of the 6 linear forms corresponding to the 
    6 side normals?
    
    Let's define the hexagon by inequalities:
    x * cos(theta_k) + y * sin(theta_k) <= R_apothem
    where theta_k are angles of normals.
    For a regular hexagon with side s, R_apothem = s * sqrt(3) / 2.
    The normals are at 0, 60, 120, 180, 240, 300 degrees?
    No, if vertices are at 0, 60..., normals are at 30, 90...
    Let's assume the outer hexagon is aligned with the inner ones initially, 
    but we can rotate it. The task allows outer rotation.
    However, due to symmetry, we can fix the outer hexagon orientation 
    and just find the scale? 
    No, the inner hexagons can rotate. 
    Actually, the optimal outer hexagon will be oriented such that its sides 
    are tangent to the "convex hull" of the inner hexagons in a specific way.
    
    Let's simplify: 
    The distance from the center of the outer hexagon to its sides is the apothem A.
    Side length s = 2 * A / sqrt(3).
    We need to find the minimum A such that all inner vertices satisfy:
    |v . n_k| <= A for all 6 normals n_k of the outer hexagon.
    This is equivalent to finding the minimum A such that:
    max_k ( max_i (v_i . n_k), max_i (-v_i . n_k) ) <= A.
    Let M_k = max( max_i (v_i . n_k), max_i (-v_i . n_k) ).
    Then A = max_k M_k.
    And s = 2 * A / sqrt(3).
    
    We can optimize the orientation of the outer hexagon too?
    The task returns (inner, center, side, angle).
    We can fix the outer center to (0,0) by translating all inner hexagons 
    to be centered around the centroid of their bounding box? 
    Actually, we can just let the optimizer place them anywhere, 
    and compute the outer hexagon centered at (0,0) with optimal orientation.
    
    Let's include the outer hexagon angle as a variable? 
    That adds 1 dimension. Total 36 + 1 = 37.
    Or we can just assume the outer hexagon is aligned with the "average" 
    or just try a few fixed orientations?
    Given the symmetry of the problem, the optimal outer hexagon likely 
    aligns with the symmetry of the packing.
    Let's stick to 36 variables (x, y, theta for 12 hexagons) 
    and compute the side length assuming the outer hexagon is aligned 
    with the global axes (angle 0). If the packing is rotated, 
    the side length calculation with fixed normals might be suboptimal.
    
    Better: Compute the "radius" in all directions? 
    No, regular hexagon has 6 specific directions.
    Let's add the outer angle as a variable. 
    Variables: x_0..x_11, y_0..y_11, theta_0..theta_11, phi (outer angle).
    Total 37 variables.
    
    Objective: Side length.
    Side length = (2/sqrt(3)) * max_{k=0..5} ( max_{i=0..11} ( max_{j=0..5} (v_{i,j} . n_k(phi)) ), 
                                              max_{i=0..11} ( max_{j=0..5} (-v_{i,j} . n_k(phi)) ) )
    Wait, the max over -v . n is the same as max over v . (-n).
    Since the set of 6 normals includes opposites, we just need:
    A(phi) = max_{k=0..5} ( max_{i,j} (v_{i,j} . n_k(phi)) )
    where n_k(phi) are the 6 unit normals.
    Then s = 2 * A(phi) / sqrt(3).
    
    This function is differentiable (max of linear functions is convex and differentiable almost everywhere).
    L-BFGS-B can handle this.
    """
    all_verts = vertices # (72, 2)
    phi = outer_angle
    # Normals of outer hexagon
    # If vertices are at 0, 60..., normals are at 30, 90...
    # Let's use normals at phi + 30, phi + 90, ...
    normal_angles = np.radians([30, 90, 150, 210, 270, 330]) + phi
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Projection of all vertices onto all normals
    # (72, 2) dot (2, 6) -> (72, 6)
    projs = all_verts @ normals.T
    
    # Max projection in each normal direction
    max_projs = np.max(projs, axis=0) # (6,)
    
    # The apothem is the max of these
    apothem = np.max(max_projs)
    
    return (2.0 / SQRT3) * apothem

def compute_contact_penalty(vertices, pos, ang, smoothness=0.05):
    """
    Computes a smooth penalty for overlaps between inner hexagons.
    For each pair (i, j), we compute the minimum distance between the two convex polygons.
    If they overlap, distance is negative (penetration depth).
    We use a smooth approximation of the minimum distance.
    
    A robust way to compute distance between two convex polygons is using the 
    Separating Axis Theorem (SAT).
    Distance = min over all axes (gap on that axis).
    If they overlap, all gaps are <= 0.
    So distance = min(gaps).
    If min_gap < 0, they overlap.
    Penalty = max(0, -min_gap)^2 ? 
    But min_gap is not differentiable at 0.
    We can use a soft min: -logsumexp(-gaps / t) * t.
    Or simply sum of squared negative gaps?
    If we use sum of squared negative gaps, it penalizes any axis where they overlap.
    This is a valid penalty for ensuring separation on all axes.
    penalty = sum_{axes} max(0, -gap_axis)^2
    
    This is differentiable.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Early exit if far apart
            dx = pos[i] - pos[j]
            dy = pos[i+1] - pos[j+1] # Wait, pos is flat array?
            # Let's pass pos and ang as separate arrays for clarity in this helper
            pass
            
    # We will implement this inside the objective function for speed.
    return 0.0

def objective(x):
    """
    x: [x0, y0, th0, x1, y1, th1, ..., x11, y11, th11, phi]
    """
    pos = x[0:2*N:3]
    y_pos = x[1:2*N:3]
    ang = x[2:3*N:3]
    phi = x[3*N]
    
    # Compute vertices
    all_verts = np.zeros((N*6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = get_vertices(pos[i], y_pos[i], ang[i])
        
    # Compute side length
    side_len = compute_outer_side_length(all_verts, phi)
    
    # Compute contact penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Get vertices for i and j
            v1 = all_verts[i*6:(i+1)*6]
            v2 = all_verts[j*6:(j+1)*6]
            
            # Compute SAT gaps
            # Axes from v1
            axes = []
            for k in range(6):
                e1 = v1[(k+1)%6] - v1[k]
                n1 = np.array([-e1[1], e1[0]])
                norm = np.linalg.norm(n1)
                if norm > 1e-10:
                    axes.append(n1 / norm)
            for k in range(6):
                e2 = v2[(k+1)%6] - v2[k]
                n2 = np.array([-e2[1], e2[0]])
                norm = np.linalg.norm(n2)
                if norm > 1e-10:
                    axes.append(n2 / norm)
            
            # Compute gaps
            min_gap = np.inf
            for n in axes:
                proj1 = v1.dot(n)
                proj2 = v2.dot(n)
                max1, min1 = np.max(proj1), np.min(proj1)
                max2, min2 = np.max(proj2), np.min(proj2)
                
                # Gap
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = 0.0
                    
                if gap < min_gap:
                    min_gap = gap
                    
            # min_gap is the distance if disjoint, 0 if touching/overlapping.
            # If they overlap, min_gap will be 0 (because on all axes they overlap).
            # Wait, if they overlap, gap is 0 for all axes. So min_gap = 0.
            # This doesn't give a negative value for overlap.
            # We need to detect overlap.
            # If min_gap == 0, check if they actually overlap.
            # If they overlap, we want a penalty.
            # A common trick: if min_gap is 0, compute the area of intersection? 
            # Or use a "penetration depth" approximation.
            
            # Alternative: Use the sum of squared distances of vertices to the other polygon?
            # If a vertex is inside, distance is 0. 
            # This is not differentiable.
            
            # Let's use a different penalty: 
            # Sum of squared "violations" of the separating axis condition.
            # If they are disjoint, there exists an axis with gap > 0.
            # If they overlap, all gaps are 0.
            # So min_gap = 0 is ambiguous.
            
            # Let's use the "soft" distance defined by:
            # d = -logsumexp(-gaps / t) * t
            # If gaps are all 0, d = -log(12 * 1) * t = -log(12)*t (negative, good for penalty?)
            # No, we want penalty > 0 when overlapping.
            
            # Let's try: Penalty = max(0, R_min - min_gap)^2
            # Where R_min is a small epsilon? No.
            
            # Let's go back to basics.
            # If min_gap > 0, they are disjoint. Penalty 0.
            # If min_gap == 0, they touch or overlap.
            # To distinguish, check if any vertex of v1 is inside v2.
            # This is not differentiable.
            
            # For L-BFGS, we need a smooth function.
            # Let's use the "L1 norm of the negative parts of the gaps"?
            # No, if they overlap, all gaps are 0.
            
            # Actually, if they overlap, the "distance" between the sets is 0.
            # We want to push them apart.
            # We can use the gradient of the distance function.
            # But distance is 0 at overlap.
            
            # Let's use a penalty based on the "overlap area" approximation?
            # Or simply: if min_gap < epsilon, add a penalty proportional to 
            # the square of the "average gap" if it's negative?
            
            # Let's use the following:
            # Compute the minimum gap.
            # If min_gap < 0.01, we assume they might be overlapping.
            # We compute the "penetration depth" as the minimum over axes of the 
            # "signed gap".
            # Signed gap = min(proj2) - max(proj1) if max(proj1) < min(proj2), 
            #             min(proj1) - max(proj2) if max(proj2) < min(proj1),
            #             0 otherwise.
            # This is what we did.
            
            # If they overlap, signed gap is 0 for all axes.
            # So we can't detect it with SAT alone without a point-in-poly check.
            
            # Let's add a point-in-poly check in the penalty calculation, 
            # but make it smooth?
            # Hard to do smoothly.
            
            # Alternative: Use a "soft" intersection metric.
            # Sum of squared distances from vertices of v1 to the edges of v2, 
            # but only if the vertex is "inside" or close?
            
            # Let's try a different approach: 
            # Minimize the side length + Lambda * Sum( max(0, -gap_ij)^2 )
            # where gap_ij is the minimum gap.
            # If they overlap, gap is 0, so penalty is 0. This is bad.
            
            # We need a penalty that is non-zero when overlapping.
            # Let's use the "area of intersection" as a proxy?
            # Or, we can use the fact that if they overlap, the center of one 
            # might be inside the other, or vertices are inside.
            
            # Let's use a "soft" point-in-poly test.
            # For a convex polygon, a point p is inside if p . n_k <= c_k for all k.
            # Let f_k(p) = p . n_k - c_k.
            # Inside if max_k f_k(p) <= 0.
            # Penalty = max(0, max_k f_k(p))^2.
            # This is smooth!
            
            # So, for each pair (i, j):
            # 1. Check if any vertex of v1 is inside v2.
            # 2. Check if any vertex of v2 is inside v1.
            # Penalty = Sum of squared "violations".
            
            # Let's implement this.
            
            # Precompute edge normals and offsets for v1 and v2?
            # v1 is fixed in this iteration.
            
            # Function to compute penalty for point p in polygon poly
            def point_in_poly_penalty(p, poly):
                # poly is (6, 2)
                # Compute normals and offsets
                # n_k is normal to edge k, pointing outwards.
                # c_k = v_k . n_k
                # Inside if p . n_k <= c_k for all k.
                # Violation = max(0, p . n_k - c_k)
                max_viol = 0.0
                for k in range(6):
                    v_k = poly[k]
                    v_next = poly[(k+1)%6]
                    e = v_next - v_k
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-10: continue
                    n = n / norm
                    # Check orientation: n should point outwards.
                    # For a CCW polygon, left normal is outwards?
                    # Let's check cross product of e and n.
                    # If cross(e, n) > 0, n is left normal.
                    # For CCW, left normal is outwards?
                    # Vertices 0..5 are CCW.
                    # Edge 0->1. Left normal points to the "left" of the edge, 
                    # which is towards the center if the polygon is CCW?
                    # No, for a CCW polygon, the interior is to the left of the edges.
                    # So the inward normal is the left normal.
                    # We want the outward normal.
                    # So we should use the right normal: [e[1], -e[0]].
                    n_out = np.array([e[1], -e[0]])
                    n_out = n_out / norm
                    c = np.dot(v_k, n_out)
                    val = np.dot(p, n_out) - c
                    if val > max_viol:
                        max_viol = val
                return max_viol ** 2
            
            # Check vertices of v1 against v2
            for v_idx in range(6):
                p = v1[v_idx]
                penalty += point_in_poly_penalty(p, v2)
                
            # Check vertices of v2 against v1
            for v_idx in range(6):
                p = v2[v_idx]
                penalty += point_in_poly_penalty(p, v1)
                
    # Scale penalty
    penalty_weight = 100.0
    return side_len + penalty_weight * penalty

def construct():
    start_time = time.time()
    
    # Initial Configuration: Honeycomb
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_x = np.zeros(3*N + 1)
    for i in range(N):
        q, r = coords[i]
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_x[3*i] = x
        init_x[3*i+1] = y
        init_x[3*i+2] = 0.0
    init_x[3*N] = 0.0 # Outer angle
    
    # Ensure validity of init
    # If not valid, scale up
    def is_valid(x):
        pos = x[0:2*N:3]
        y_pos = x[1:2*N:3]
        ang = x[2:3*N:3]
        for i in range(N):
            for j in range(i+1, N):
                v1 = get_vertices(pos[i], y_pos[i], ang[i])
                v2 = get_vertices(pos[j], y_pos[j], ang[j])
                # Simple check: if centers are too close, likely overlap
                # But let's use the point-in-poly check
                # Check if any vertex of v1 is inside v2
                for v_idx in range(6):
                    p = v1[v_idx]
                    inside = True
                    for k in range(6):
                        vk = v2[k]
                        vn = v2[(k+1)%6]
                        e = vn - vk
                        n_out = np.array([e[1], -e[0]])
                        norm = np.linalg.norm(n_out)
                        if norm < 1e-10: continue
                        n_out /= norm
                        c = np.dot(vk, n_out)
                        if np.dot(p, n_out) > c + 1e-6:
                            inside = False
                            break
                    if inside:
                        return False
        return True

    scale = 1.0
    while not is_valid(init_x) and scale < 5.0:
        init_x[0:2*N:3] *= scale
        init_x[1:2*N:3] *= scale
        scale += 0.1
        
    # Optimize with L-BFGS-B
    best_x = init_x.copy()
    best_val = objective(best_x)
    
    try:
        res = minimize(objective, init_x, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6})
        
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception as e:
        pass # Keep best_x
        
    # Refine with a few more steps or a second run?
    if time.time() - start_time < TIME_LIMIT - 10:
        # Perturb best_x
        x2 = best_x + np.random.normal(0, 0.1, 3*N + 1)
        try:
            res2 = minimize(objective, x2, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except:
            pass

    # Extract result
    inner = []
    for i in range(N):
        x = float(best_x[3*i])
        y = float(best_x[3*i+1])
        ang = float(np.degrees(best_x[3*i+2])) % 360.0
        inner.append([x, y, ang])
        
    # Recompute side length and center
    pos = best_x[0:2*N:3]
    y_pos = best_x[1:2*N:3]
    ang_rad = best_x[2:3*N:3]
    phi = best_x[3*N]
    
    all_verts = np.zeros((N*6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = get_vertices(pos[i], y_pos[i], ang_rad[i])
        
    # The outer hexagon is centered at (0,0) in our calculation.
    # But the inner hexagons might be shifted.
    # We should center the inner hexagons so that their "bounding box" is centered at (0,0)
    # to minimize the side length?
    # Actually, the objective function assumed the outer hexagon is centered at (0,0).
    # If the inner hexagons are shifted, the side length will be larger.
    # We should translate the inner hexagons so that the "center of mass" of the 
    # support function is at (0,0).
    # Or simply, find the optimal translation that minimizes the side length.
    
    # Let's compute the side length for the current config, 
    # but also try to center it.
    
    # The side length is determined by max projection.
    # If we translate all inner hexagons by t, the projections shift by t . n_k.
    # We want to minimize max_k ( max_i (v_i . n_k) + t . n_k ).
    # This is a convex optimization problem for t.
    # But for simplicity, we can just center the vertices.
    
    center_x = np.mean(all_verts[:, 0])
    center_y = np.mean(all_verts[:, 1])
    
    # Shift all inner hexagons by -center
    for i in range(N):
        inner[i][0] -= center_x
        inner[i][1] -= center_y
        
    # Recompute vertices with new centers
    all_verts_centered = np.zeros((N*6, 2))
    for i in range(N):
        all_verts_centered[i*6:(i+1)*6] = get_vertices(inner[i][0], inner[i][1], ang_rad[i])
        
    # Compute side length
    side_len = compute_outer_side_length(all_verts_centered, phi)
    
    # Return
    return inner, (0.0, 0.0), float(side_len), float(np.degrees(phi)) % 360.0

# EVOLVE-BLOCK-END
