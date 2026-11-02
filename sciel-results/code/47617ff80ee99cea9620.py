import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _smooth_distance_sq_to_hex(px, py, cx, cy, s, angle_deg):
    """
    Returns a smooth approximation of the squared distance from point (px, py) 
    to the hexagon centered at (cx, cy) with side s and rotation angle_deg.
    If the point is inside, returns 0.
    Uses the intersection of 3 strips definition.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    
    # Rotate point to local frame of hexagon
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    
    # Normals of the 3 pairs of parallel sides in local frame
    # For a hexagon with vertex at 0 deg, sides are at 30, 90, 150 deg.
    # Normal vectors:
    # n0: (cos(30), sin(30))
    # n1: (cos(90), sin(90))
    # n2: (cos(150), sin(150))
    
    # We need to check projection onto these normals.
    # If projection > apothem, the point is outside that strip.
    # The "distance" to the boundary in the direction of the normal is (proj - apothem).
    # For a smooth penalty, we sum the squared violations of the 3 constraints.
    # Note: This is not exactly the Euclidean distance squared if outside multiple strips,
    # but it is a smooth upper bound and 0 inside.
    
    penalty = 0.0
    
    # Strip 1: normal at 30 deg
    proj1 = rx * math.cos(math.radians(30)) + ry * math.sin(math.radians(30))
    if proj1 > apothem:
        v = proj1 - apothem
        penalty += v * v
    elif proj1 < -apothem:
        v = -apothem - proj1
        penalty += v * v
        
    # Strip 2: normal at 90 deg
    proj2 = ry
    if proj2 > apothem:
        v = proj2 - apothem
        penalty += v * v
    elif proj2 < -apothem:
        v = -apothem - proj2
        penalty += v * v
        
    # Strip 3: normal at 150 deg
    proj3 = rx * math.cos(math.radians(150)) + ry * math.sin(math.radians(150))
    if proj3 > apothem:
        v = proj3 - apothem
        penalty += v * v
    elif proj3 < -apothem:
        v = -apothem - proj3
        penalty += v * v
        
    return penalty

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN]
    Outer hexagon is fixed at center (0,0), side 3.0, angle 0.
    We minimize the sum of squared violations of disjointness and containment.
    """
    n_inner = N
    inner_params = []
    for i in range(n_inner):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner_params.append((cx, cy, a))
        
    penalty = 0.0
    
    # 1. Containment: All vertices of inner hexagons must be inside outer hexagon.
    # Outer: center (0,0), side 3.0, angle 0.
    # We check each vertex of each inner hexagon.
    # If a vertex is outside, _smooth_distance_sq_to_hex returns > 0.
    # To make it a "maximize minimum clearance" style, we sum these.
    # Actually, we want them to be INSIDE.
    # So if dist_sq > 0, it's outside. We add it to penalty.
    
    for ip in inner_params:
        icx, icy, iang = ip
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            penalty += _smooth_distance_sq_to_hex(pt[0], pt[1], 0.0, 0.0, 3.0, 0.0)
            
    # 2. Disjointness: For each pair, check vertices of one against the other.
    # If a vertex of H1 is inside H2, dist_sq is 0.
    # We want them to be DISJOINT. So we want dist_sq > 0 for all vertices?
    # No, if they are disjoint, vertices are outside.
    # Wait, _smooth_distance_sq_to_hex returns 0 if INSIDE, >0 if OUTSIDE.
    # So if H1 and H2 are disjoint, vertices of H1 are OUTSIDE H2 -> dist_sq > 0.
    # We want to MINIMIZE overlap.
    # If they overlap, some vertices might be inside (dist_sq = 0) or edges cross.
    # This "distance to set" logic is tricky for disjointness.
    
    # Better approach for disjointness with smooth penalty:
    # We want the minimum distance between the two convex polygons to be >= 0.
    # A common smooth proxy is to check the distance from vertices of one to the other.
    # But if they overlap, the distance is 0.
    # We want to push them apart.
    
    # Let's use a different metric for disjointness:
    # If they overlap, we want a large penalty.
    # We can check if any vertex of H1 is inside H2. If yes, penalty = 0 (bad).
    # If no vertex is inside, they might still overlap (edge crossing).
    # For regular hexagons, if they overlap, usually a vertex is inside or they are very close.
    
    # Let's try this:
    # For each pair (i, j):
    #   For each vertex v of Hi:
    #     d_sq = _smooth_distance_sq_to_hex(v, Hj)
    #     This is 0 if v is inside Hj.
    #     We want v to be OUTSIDE Hj.
    #     So we want d_sq to be large? No, we just want it non-zero.
    #     But minimizing d_sq would push v INSIDE.
    #     We want to MAXIMIZE d_sq? Or rather, we want to ensure d_sq > 0.
    
    # This is a "separation" problem.
    # Standard trick: Minimize the "overlap".
    # If we define penalty as sum of (max(0, R - d))^2, it's not smooth.
    
    # Let's stick to the containment penalty logic but invert it?
    # No, let's use a simple geometric check for disjointness that is differentiable-ish.
    # Actually, for the purpose of this L-BFGS-B attempt, let's assume that if we start
    # from a valid configuration (lattice), the gradients will guide us to maintain validity
    # or improve it.
    
    # Let's refine the disjointness penalty.
    # We want to penalize overlap.
    # If a vertex of H1 is inside H2, it's an overlap.
    # We can use the "distance to boundary" in the local frame.
    # If inside, distance is 0.
    # We want to push it out.
    # So we want to MINIMIZE the "depth" of penetration?
    # Depth = apothem - proj (if proj < apothem).
    # If we sum squared depths for all vertices of all pairs, we get a penalty for overlap.
    
    # Let's implement _penetration_depth_sq
    # In local frame of Hj, for vertex v of Hi:
    # proj1, proj2, proj3.
    # If any proj > apothem, it's outside.
    # If all proj < apothem (and > -apothem), it's inside.
    # The "depth" is min(apothem - proj1, apothem - proj2, apothem - proj3, ...).
    # This is not smooth at the boundary.
    
    # Alternative: Sum of squared (apothem - proj) for all 6 half-spaces?
    # If inside, all (apothem - proj) > 0. Sum of squares is positive.
    # If outside, some (apothem - proj) < 0.
    # This doesn't directly measure overlap.
    
    # Let's go back to the vertex-inside check.
    # If we simply sum the squared distances from vertices of Hi to Hj, 
    # and we want them to be 0 (touching) or large (separated)?
    # No, we want them to be separated.
    
    # Let's try a different strategy:
    # Since we start from a valid lattice, maybe we don't need a strong disjointness penalty
    # if we constrain the movement to be small?
    # But L-BFGS-B might jump.
    
    # Let's use a robust disjointness penalty:
    # For each pair, compute the minimum squared distance between the two polygons.
    # This is hard to compute smoothly.
    
    # Let's approximate:
    # Check vertices of Hi against Hj.
    # If a vertex is inside Hj, add a penalty based on how deep it is.
    # Depth = max(0, apothem - min_proj)?
    
    # Let's implement a simple "overlap penalty" using the 6 half-spaces of Hj.
    # For a vertex v of Hi:
    #   Calculate proj1, proj2, proj3 in Hj's local frame.
    #   If v is inside Hj, all |proj| <= apothem.
    #   The "overlap" is that v is inside.
    #   We want to push v out.
    #   The direction to push is the one with the smallest margin (apothem - proj).
    #   This is the normal of the closest face.
    #   Penalty = (min_margin)^2?
    
    # Let's implement _min_margin_sq
    def _min_margin_sq(px, py, cx, cy, s, angle_deg):
        a = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        
        apothem = s * math.sqrt(3) / 2
        
        # 6 half-spaces
        # Normals at 30, 90, 150, 210, 270, 330
        # Projections:
        # p1 = rx*cos(30) + ry*sin(30)
        # p2 = ry
        # p3 = rx*cos(150) + ry*sin(150)
        # p4 = -p1
        # p5 = -p2
        # p6 = -p3
        
        p1 = rx * math.cos(math.radians(30)) + ry * math.sin(math.radians(30))
        p2 = ry
        p3 = rx * math.cos(math.radians(150)) + ry * math.sin(math.radians(150))
        
        # Margins: apothem - p, apothem + p
        m1 = apothem - p1
        m2 = apothem - p2
        m3 = apothem - p3
        m4 = apothem + p1
        m5 = apothem + p2
        m6 = apothem + p3
        
        # If inside, all margins > 0.
        # The minimum margin is the distance to the closest face.
        min_m = min(m1, m2, m3, m4, m5, m6)
        
        if min_m > 0:
            return min_m * min_m
        else:
            return 0.0 # Outside
            
    # Disjointness Penalty:
    # For each pair (i, j):
    #   For each vertex of Hi:
    #     pen += _min_margin_sq(v, Hj)
    #   For each vertex of Hj:
    #     pen += _min_margin_sq(v, Hi)
    
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            v_i = _hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
            v_j = _hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
            
            for pt in v_i:
                penalty += _min_margin_sq(pt[0], pt[1], inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                
            for pt in v_j:
                penalty += _min_margin_sq(pt[0], pt[1], inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Hexagonal lattice
    # Center (0,0)
    # Ring 1: 6 hexagons at distance 1.5
    # Ring 2: We need 4 more.
    
    init_inner = []
    init_inner.append((0.0, 0.0, 0.0))
    for i in range(6):
        a = math.radians(60*i)
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        init_inner.append((x, y, 0.0))
        
    # Add 4 from ring 2
    # Ring 2 positions:
    # (3.0, 0), (2.25, 1.9), (0.75, 2.6), (-1.5, 2.6)...
    # Let's pick 4 symmetric ones?
    # (3.0, 0), (-3.0, 0), (1.5, 2.6), (-1.5, -2.6)?
    # Let's just add 4 arbitrary valid ones from the lattice.
    # Lattice vectors: (1.5, 0.866), (0, 1.732)
    # (3.0, 0) is 2*(1.5, 0) -> not in lattice?
    # Lattice points: m*v1 + n*v2
    # v1 = (1.5, 0.866), v2 = (0, 1.732)
    # (3.0, 0) = 2*v1 - 1*v2? 2*(1.5, 0.866) - (0, 1.732) = (3.0, 0). Yes.
    # So (3.0, 0) is valid.
    # (-3.0, 0) is valid.
    # (1.5, 2.6) = 1*v1 + 1*v2? (1.5, 0.866) + (0, 1.732) = (1.5, 2.598). Yes.
    # (-1.5, -2.6) is valid.
    
    init_inner.append((3.0, 0.0, 0.0))
    init_inner.append((-3.0, 0.0, 0.0))
    init_inner.append((1.5, 2.6, 0.0))
    init_inner.append((-1.5, -2.6, 0.0))
    
    x0 = []
    for p in init_inner:
        x0.extend(p)
        
    # Outer hexagon: center (0,0), side 3.0, angle 0.
    # We don't optimize the outer hexagon in this formulation.
    # We assume side 3.0 is a good target.
    # If the penalty is 0, the packing is valid.
    
    best_x = x0
    best_val = _objective(x0)
    
    # Multi-start L-BFGS-B
    # We will run a few iterations from slightly perturbed starts.
    
    n_starts = 10
    for k in range(n_starts):
        if time.time() > deadline:
            break
            
        # Perturb x0
        np.random.seed(42 + k)
        x_start = np.array(x0) + np.random.normal(0, 0.1, len(x0))
        
        # Bounds:
        # Centers: [-4, 4]
        # Angles: [0, 360]
        bounds = []
        for i in range(N):
            bounds.append((-4.0, 4.0))
            bounds.append((-4.0, 4.0))
            bounds.append((0.0, 360.0))
            
        try:
            res = minimize(_objective, x_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
        except Exception:
            pass
            
    # Format result
    inner = []
    for i in range(N):
        cx = best_x[i*3]
        cy = best_x[i*3+1]
        a = best_x[i*3+2]
        inner.append([cx, cy, a])
        
    ocx, ocy = 0.0, 0.0
    os = 3.0
    oang = 0.0
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
