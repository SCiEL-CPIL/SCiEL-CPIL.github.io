# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Time budget
T_START = time.time()
TIME_LIMIT = 580  # Leave 50s margin before 630s kill

def _hex_vertices(cx, cy, s, a_deg):
    """Return (6,2) array of vertices for a regular hexagon."""
    a = math.radians(a_deg)
    angles = a + np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals_hex(vertices):
    """Return (6,2) array of outward-pointing unit normals for each edge of a hexagon."""
    # vertices are in order, so edge i is from vertices[i] to vertices[(i+1)%6]
    edges = np.roll(vertices, -1, axis=0) - vertices
    # Outward normal: for a convex polygon with vertices in CCW order,
    # the outward normal of an edge (dx, dy) is (dy, -dx) normalized.
    # Let's verify orientation. Standard hexagon with a=0:
    # v0=(1,0), v1=(0.5, .866). Edge v0->v1 = (-0.5, .866).
    # Normal (dy, -dx) = (.866, 0.5). This points away from center (0,0). Correct.
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms

def _min_gap(v1, n1, v2, n2):
    """
    Compute the minimum gap between two convex polygons given their vertices and normals.
    Returns (min_gap, bool_separated).
    If separated, min_gap > 0. If touching/intersecting, min_gap <= 0.
    """
    min_gap = np.inf
    # Check all edges of poly 1
    for i in range(len(v1)):
        n = n1[i]
        p = v1[i]
        # Project poly 2 onto this normal
        proj = np.dot(v2 - p, n)
        # If all projections are positive, they are separated in this direction
        # The gap is the minimum projection
        if np.min(proj) > 0:
            min_gap = min(min_gap, np.min(proj))
            # If we find a separating axis, we can potentially stop early if we want,
            # but we need the MINIMUM gap to determine the "separation" for the objective.
            # Actually, for validity we just need *some* axis with gap > 1e-6.
            # For optimization, we want to maximize the minimum gap over all pairs.
            # The "gap" between two convex bodies is the maximum over all separating axes of the distance.
            # Wait. The distance between two convex sets A and B is min_{a in A, b in B} ||a-b||.
            # This is equivalent to max over all unit vectors u of (min_{b in B} u.b - max_{a in A} u.a) if positive.
            # So for each normal, we calculate the separation.
            # The gap is the MAXIMUM of these separations (if they are separated).
            # If they are not separated, the max separation is <= 0.
            # So we want to MAXIMIZE this value for each pair.
            # If the max separation is > 0, they are disjoint.
            # If the max separation is <= 0, they intersect.
            # So for a pair, gap = max(0, max_axis_separation).
            # We want to maximize the minimum gap over all pairs.
    # Check all edges of poly 2
    for i in range(len(v2)):
        n = n2[i]
        p = v2[i]
        proj = np.dot(v1 - p, n)
        if np.min(proj) > 0:
            min_gap = min(min_gap, np.min(proj)) # Wait, this logic is wrong.
    
    # Correct Logic for Gap:
    # The distance between two convex polygons is the maximum value of d such that
    # there exists a unit vector u where min_{b in B} u.(b-c) - max_{a in A} u.(a-c) >= d.
    # Actually, simpler:
    # For each edge normal u of A or B:
    #   proj_A = dot(A, u), proj_B = dot(B, u)
    #   sep = min(proj_B) - max(proj_A)  (if u points from A to B)
    #   OR sep = min(proj_A) - max(proj_B) (if u points from B to A)
    #   Since normals are outward, for edge i of A, n1[i] points away from A.
    #   So we check if B is "outside" A in direction n1[i].
    #   sep = min(dot(B - A_i, n1[i])) where A_i is vertex on edge i.
    #   If sep > 0, B is separated from A by this edge.
    #   The gap is the MAXIMUM of all such positive separations? No.
    #   The gap is the minimum distance between any point in A and any point in B.
    #   This is equal to the maximum over all separating axes of the signed distance.
    #   If they overlap, all signed distances are <= 0, so max is <= 0.
    #   If they are separated, there is at least one axis with positive distance.
    #   The actual Euclidean distance is the maximum of these signed distances.
    
    max_sep = -np.inf
    # Normals of 1
    for i in range(len(v1)):
        n = n1[i]
        p = v1[i]
        # Distance from p to B along n
        # We want the minimum dot product of (v2 - p) with n.
        # If this min is positive, B is entirely in the half-plane defined by edge i of 1, outside 1.
        dist = np.min(np.dot(v2 - p, n))
        if dist > max_sep:
            max_sep = dist
            
    # Normals of 2
    for i in range(len(v2)):
        n = n2[i]
        p = v2[i]
        dist = np.min(np.dot(v1 - p, n))
        if dist > max_sep:
            max_sep = dist
            
    return max_sep

def _is_valid(inner, outer, side):
    """
    Check validity of the packing.
    inner: (11, 3) array of (x, y, angle)
    outer: (3,) array of (x, y, angle)
    side: float
    """
    # 1. Finite values
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if not np.isfinite(side) or side < math.sqrt(11):
        return False

    # Shift coordinates so outer center is at origin
    ocx, ocy, oang = outer
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= ocx
    inner_shifted[:, 1] -= ocy

    # Outer hexagon vertices
    outer_verts = _hex_vertices(0, 0, side, oang)
    outer_normals = _edge_normals_hex(outer_verts)

    # Inner hexagons data
    inner_verts = []
    inner_normals = []
    for i in range(11):
        cx, cy, ang = inner_shifted[i]
        v = _hex_vertices(cx, cy, 1.0, ang)
        n = _edge_normals_hex(v)
        inner_verts.append(v)
        inner_normals.append(n)
        
        # Check if inner hexagon is inside outer hexagon
        # For each outer edge, all inner vertices must be on the inside.
        # Inside means dot(v - p1, normal) <= 0 ?
        # Normal is outward. So dot(v - p1, normal) <= 0 means v is on the inside or on the edge.
        # The task says: cross(p2 - p1, p - p1) >= -1e-6.
        # Let's stick to the task's cross product check to be safe, or use the normal check which is equivalent for convex polygons.
        # Normal check: dot(v - p_edge_start, n_edge) <= 1e-6 (allowing tiny epsilon)
        # Actually, the task says "cross(p2 - p1, p - p1) >= -1e-6".
        # For a CCW polygon, cross(p2-p1, p-p1) > 0 means p is to the left of the edge (inside).
        # So cross >= -1e-6 allows p to be slightly outside or on the edge.
        # This is equivalent to dot(v - p1, n) <= epsilon if n is the outward normal?
        # Let's verify. Edge e = p2 - p1. Outward normal n = (e.y, -e.x)/|e|.
        # cross(e, v - p1) = e.x*(v-p1).y - e.y*(v-p1).x
        # dot(v - p1, n) = (e.y*(v-p1).x - e.x*(v-p1).y) / |e| = -cross(e, v-p1)/|e|.
        # So cross(e, v-p1) >= -1e-6  <=>  -dot(v-p1, n)*|e| >= -1e-6  <=>  dot(v-p1, n) <= 1e-6/|e|.
        # Since |e|=1 for unit hexagons? No, outer hexagon side is 'side'. So |e|=side.
        # So dot(v-p1, n) <= 1e-6/side.
        
        for j in range(6):
            p1 = outer_verts[j]
            n = outer_normals[j]
            # dot(inner_vert - p1, n) should be <= 1e-6/side
            dists = np.dot(v - p1, n)
            if np.max(dists) > 1e-6 / side:
                return False

    # Check separation between all pairs of inner hexagons
    for i in range(11):
        for j in range(i + 1, 11):
            gap = _min_gap(inner_verts[i], inner_normals[i], inner_verts[j], inner_normals[j])
            # "separated: on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
            # This means the gap must be > 1e-6.
            if gap <= 1e-6:
                return False
                
    # Check edge lengths of shifted inner hexagons
    for i in range(11):
        v = inner_verts[i]
        for k in range(6):
            e = v[(k+1)%6] - v[k]
            length = np.linalg.norm(e)
            if abs(length - 1.0) > 1e-8:
                return False

    return True

def _objective(params, initial_cluster):
    """
    Objective function for optimization.
    params: [rotation_angle, ... maybe others if we allow deformation? No, rigid cluster.]
    For now, just rotation.
    Later, we might optimize the cluster shape too.
    """
    t = time.time()
    if t - T_START > TIME_LIMIT - 10:
        return 1e10

    rot = params[0]
    
    # Apply rotation to the initial cluster
    # initial_cluster is (11, 3) array of (x, y, angle)
    cluster = initial_cluster.copy()
    cluster[:, 0:2] = np.column_stack([
        cluster[:, 0] * math.cos(rot) - cluster[:, 1] * math.sin(rot),
        cluster[:, 0] * math.sin(rot) + cluster[:, 1] * math.cos(rot)
    ])
    cluster[:, 2] += np.degrees(rot)

    # Compute bounding box / min enclosing hexagon side
    # To find the min side of a regular hexagon enclosing this cluster:
    # We can try rotating the outer hexagon as well.
    # The side length S is determined by the max projection of the cluster vertices onto the 6 directions of the outer hexagon.
    # Specifically, if the outer hexagon has angle A, its edges have normals at A, A+60, etc.
    # The distance from center to each edge is S * cos(30) = S * sqrt(3)/2.
    # So for each normal direction n_k, max(dot(cluster_verts, n_k)) <= S * sqrt(3)/2.
    # Thus S >= 2/sqrt(3) * max_k (max_dot_k).
    # We want to minimize S. This means minimizing max_k (max_dot_k) over all 6 directions.
    # And we can choose the outer hexagon angle A to minimize this max.
    
    # Let's collect all vertices of the cluster
    all_verts = []
    for i in range(11):
        v = _hex_vertices(cluster[i, 0], cluster[i, 1], 1.0, cluster[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # Center the cluster? The outer hexagon can be centered anywhere.
    # Optimal center is likely the centroid or center of the bounding box.
    # For a regular hexagon, the center that minimizes the max projection is the center of the minimum enclosing circle?
    # Actually, for a fixed set of points, the minimum enclosing regular hexagon can be found by rotating the hexagon.
    # Let's assume the center is the centroid for now, or optimize it too.
    # Let's just use the centroid.
    cx, cy = np.mean(all_verts[:, 0]), np.mean(all_verts[:, 1])
    centered_verts = all_verts - np.array([cx, cy])
    
    # Now find min S by rotating outer hexagon
    # We can sample outer hexagon angles or use optimization.
    # Since we are already in an optimizer, let's just calculate the required S for a given outer angle.
    # But we want to minimize S.
    # Let's define a helper to calc S for a given outer angle.
    
    def calc_side(outer_angle):
        a = math.radians(outer_angle)
        # Normals of outer hexagon
        angles = a + np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_proj = 0
        for n in normals:
            proj = np.dot(centered_verts, n)
            # We need max(proj) and min(proj) to center it?
            # No, we centered at centroid. The hexagon is centered at origin.
            # The condition is max(proj) <= S*sqrt(3)/2 AND min(proj) >= -S*sqrt(3)/2.
            # So S >= 2/sqrt(3) * max(max(proj), -min(proj)).
            m = max(np.max(proj), -np.min(proj))
            if m > max_proj:
                max_proj = m
        return 2.0 / math.sqrt(3.0) * max_proj

    # Minimize calc_side(outer_angle) over outer_angle in [0, 60)
    # We can do a simple scan or use scipy.
    # To be fast, let's do a coarse scan and then refine.
    best_s = 1e10
    best_ao = 0
    # Coarse scan
    for ao in np.linspace(0, 60, 61):
        s = calc_side(ao)
        if s < best_s:
            best_s = s
            best_ao = ao
            
    # Refine
    res = minimize(calc_side, best_ao, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 50})
    if res.success:
        best_s = res.fun
        best_ao = res.x[0]
        
    # Now check validity of this configuration
    # We need to construct the full inner and outer arrays
    inner_final = cluster.copy()
    outer_final = np.array([cx, cy, best_ao])
    
    if _is_valid(inner_final, outer_final, best_s):
        return 1.0 / best_s
    else:
        return 1e10

def _optimize_cluster(cluster_params):
    """
    Optimize the rigid cluster.
    cluster_params: (11, 3) array.
    """
    # This function will be called by a higher-level optimizer?
    # Or we can just optimize the rotation of a fixed cluster first.
    pass

def hexagon_packing_11():
    # Initial cluster: a tight packing of 11 hexagons.
    # One common dense packing is a hexagonal number configuration: 1 in center, 6 around, 4 more?
    # Or just a 3x3 grid with corners removed?
    # Let's try a "flower" with 7, and add 4 more around.
    # Hexagon side 1. Distance between centers of touching hexagons is 2*sqrt(3)/2 = sqrt(3)? No.
    # Distance between centers of touching unit hexagons is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3).
    # Wait, apothem of unit hexagon is sqrt(3)/2.
    # So centers are sqrt(3) apart.
    
    # Let's define a base cluster of 7 hexagons (center + 6 neighbors).
    # Center at (0,0).
    # Neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300.
    # But they must not overlap.
    # If all have angle 0, they touch.
    # Let's generate this cluster.
    
    d = math.sqrt(3)
    base = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [d/2, d*math.sqrt(3)/2, 0],
        [-d/2, d*math.sqrt(3)/2, 0],
        [-d, 0, 0],
        [-d/2, -d*math.sqrt(3)/2, 0],
        [d/2, -d*math.sqrt(3)/2, 0]
    ])
    
    # Add 4 more hexagons. Where to put them?
    # Maybe in the "gaps" of the flower?
    # The gaps are at distance d * 2 / sqrt(3)? No.
    # Let's just place them further out.
    # Or maybe a different cluster shape is better.
    # Let's try a 3x3 grid of hexagons (9 total) + 2 more?
    # Grid spacing: x = 1.5, y = sqrt(3)/2.
    # For flat-top hexagons (angle 0):
    # Row 0: x = 0, 1.5, 3.0
    # Row 1: x = 0.75, 2.25 (shifted)
    # Row 2: x = 0, 1.5, 3.0
    # This is 8 hexagons.
    # Let's use a 3x3 grid centered at origin.
    # x: -1.5, 0, 1.5
    # y: -sqrt(3)/2, 0, sqrt(3)/2
    # But rows are shifted.
    # Row y=0: x = -1.5, 0, 1.5
    # Row y=h: x = -0.75, 0.75
    # Row y=-h: x = -0.75, 0.75
    # Total 7.
    
    # Let's stick with the 7-flower and add 4 more at the "points" of the star.
    # Points of the star are at distance 2*d?
    # Let's just use a random initial cluster and let the optimizer fix it?
    # No, rigid cluster means we fix the relative positions.
    # We need a good initial cluster.
    
    # Let's try to construct a cluster of 11 hexagons that is known to be compact.
    # Reference: "Packing 11 hexagons in a hexagon".
    # Often, the optimal packing involves some hexagons rotated.
    
    # For now, let's just optimize the rotation of the 7-flower + 4 added at the end of the axes.
    # Added hexagons at:
    # (2*d, 0), (2*d*cos(60), 2*d*sin(60)), etc.
    # This makes a larger flower.
    
    extra = np.array([
        [2*d, 0, 0],
        [d, d*math.sqrt(3), 0],
        [-d, d*math.sqrt(3), 0],
        [-2*d, 0, 0]
    ])
    # Wait, that's only 4. Total 11.
    # But do they overlap?
    # Distance from (2*d, 0) to (d, 0) is d. They touch.
    # Distance from (2*d, 0) to (d/2, d*sqrt(3)/2) is sqrt((1.5d)^2 + (d*sqrt(3)/2)^2) = d*sqrt(2.25 + 0.75) = d*sqrt(3) = 3.
    # Distance d is sqrt(3) ~ 1.732.
    # So they are far apart.
    
    cluster = np.vstack([base, extra])
    
    # Optimize the rotation of this cluster
    bounds = [(0, 2*math.pi)]
    res = differential_evolution(_objective, bounds, args=(cluster,), maxiter=50, popsize=10, seed=42, polish=False)
    
    best_rot = res.x[0]
    best_val = res.fun
    
    # Construct the final result
    cluster_final = cluster.copy()
    cluster_final[:, 0:2] = np.column_stack([
        cluster_final[:, 0] * math.cos(best_rot) - cluster_final[:, 1] * math.sin(best_rot),
        cluster_final[:, 0] * math.sin(best_rot) + cluster_final[:, 1] * math.cos(best_rot)
    ])
    cluster_final[:, 2] += np.degrees(best_rot)
    
    # Recalculate best side and outer angle
    all_verts = []
    for i in range(11):
        v = _hex_vertices(cluster_final[i, 0], cluster_final[i, 1], 1.0, cluster_final[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    cx, cy = np.mean(all_verts[:, 0]), np.mean(all_verts[:, 1])
    centered_verts = all_verts - np.array([cx, cy])
    
    def calc_side(outer_angle):
        a = math.radians(outer_angle)
        angles = a + np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        max_proj = 0
        for n in normals:
            proj = np.dot(centered_verts, n)
            m = max(np.max(proj), -np.min(proj))
            if m > max_proj:
                max_proj = m
        return 2.0 / math.sqrt(3.0) * max_proj

    best_s = 1e10
    best_ao = 0
    for ao in np.linspace(0, 60, 61):
        s = calc_side(ao)
        if s < best_s:
            best_s = s
            best_ao = ao
            
    res2 = minimize(calc_side, best_ao, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 50})
    if res2.success:
        best_s = res2.fun
        best_ao = res2.x[0]
        
    outer_final = np.array([cx, cy, best_ao])
    
    # Final validity check
    if _is_valid(cluster_final, outer_final, best_s):
        return cluster_final, outer_final, best_s
    else:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0]
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
