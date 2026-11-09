"""
Optimized hexagon packing using Differentiable SDF and L-BFGS-B.
Strategy: 
1. Represent inner hexagons via Signed Distance Fields (SDF) in local coordinates.
2. Use L-BFGS-B to minimize outer side length + smooth overlap/boundary penalties.
3. Gradients are computed analytically for the SDF terms, providing a smooth landscape
   unlike the discrete SAT checks in the previous approach.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_sdf_local(x, y):
    """
    Compute SDF for a unit regular hexagon (side 1) centered at origin, 
    aligned with axes (flat top/bottom? or pointy? 
    Standard math: vertices at 0, 60, ... degrees.
    For side length 1, circumradius R=1, inradius r=sqrt(3)/2.
    The SDF of a convex polygon can be computed by finding the closest point on the boundary.
    However, for a regular hexagon, we can use a simpler analytic approximation or exact projection.
    
    Let's use the standard SDF for a regular hexagon.
    A regular hexagon with side 1 has vertices at:
    (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    The distance to the boundary is the minimum distance to the 6 edges.
    Inside: negative, Outside: positive.
    """
    # Due to symmetry, we can map (x,y) to the first sector [0, 30 degrees]
    # and compute distance to the edge in that sector.
    # But simpler: iterate edges. For performance in Python loops, this is slow.
    # We need a fast SDF.
    
    # Alternative: Use the fact that hexagon is intersection of 6 half-planes.
    # d(x,y) = min over 6 edges of (signed distance to line)
    # If point is inside all, it's negative.
    
    # Vertices:
    # v0: (1, 0)
    # v1: (0.5, h)
    # v2: (-0.5, h)
    # v3: (-1, 0)
    # v4: (-0.5, -h)
    # v5: (0.5, -h)
    # where h = sqrt(3)/2
    
    # Edges:
    # e0: v0->v1. Line: -sqrt(3)/2 x + 1/2 y = 3/4 ? No.
    # Let's just use the analytic formula for distance to regular hexagon.
    # From "Signed Distance Functions for Polygons" or similar.
    
    # For a regular hexagon with inradius r = sqrt(3)/2 and circumradius R=1:
    # The SDF can be computed by:
    # 1. Map to first 60 deg sector.
    # 2. If angle is in [0, 30], distance to edge v0-v1.
    # 3. If angle in [30, 60], distance to edge v1-v2.
    # By symmetry, we can reduce to [0, 30] degrees.
    
    h = math.sqrt(3) / 2.0
    
    # Rotate to align with symmetry axis?
    # Let's just compute distance to the 6 edges directly.
    # Edge 0: from (1,0) to (0.5, h). Vector: (-0.5, h). Normal: (h, 0.5) normalized?
    # Length of edge is 1. Normal is (h, 0.5). Wait, (-0.5, h) dot (h, 0.5) = -0.5h + 0.5h = 0.
    # Norm of (h, 0.5) is sqrt(3/4 + 1/4) = 1. So normal is (h, 0.5).
    # Distance to line containing edge 0: (x - 1)*h + (y - 0)*0.5
    # But we must check if projection falls on the segment.
    
    # This is getting complicated for a fast Python function.
    # Let's use a simpler approximation or a library if available? No, standard lib only.
    
    # Let's use the "closest point on polygon" logic but vectorized if possible? 
    # No, we are in a gradient step, so we need derivatives.
    
    # Actually, for L-BFGS, we can just use the "soft" minimum of distances to edges.
    # Or, better, we can use the fact that for packing, we mostly care about the "gap".
    
    # Let's try a different approach: 
    # Minimize the maximum of the "support functions" for the outer hexagon, 
    # and use a smooth penalty for overlaps based on center distance and angle, 
    # but that was the old way.
    
    # The prompt asks for SDF.
    # Let's implement a fast SDF for unit hexagon.
    
    # Reference: 
    # def sdHex(p, r):
    #     p = abs(p)
    #     p = max(p.x - p.y, p.x)
    #     ...
    
    # For a regular hexagon with side 1:
    # The SDF is:
    # d = max( |x| - 1, |y| - h, |x/2 + y*sqrt(3)/2| - 0.5, |x/2 - y*sqrt(3)/2| - 0.5 )
    # Wait, that's for the intersection of strips.
    # The hexagon is the intersection of 6 half planes.
    # The SDF of an intersection is the max of the SDFs of the half planes.
    # SDF of half plane (n, d) is n.p - d.
    
    # Half planes for unit hexagon (vertices at 0, 60...):
    # 1. x <= 1
    # 2. x/2 + y*sqrt(3)/2 <= sqrt(3)/2  => x + y*sqrt(3) <= sqrt(3)
    # 3. -x/2 + y*sqrt(3)/2 <= sqrt(3)/2 => -x + y*sqrt(3) <= sqrt(3)
    # 4. x >= -1
    # 5. -x - y*sqrt(3) <= sqrt(3)
    # 6. x - y*sqrt(3) <= sqrt(3)
    
    # Let's verify vertex (1,0):
    # 1. 1 <= 1 (0)
    # 2. 1 + 0 <= 1.732 (0.732)
    # 3. -1 + 0 <= 1.732 (2.732)
    # Max is 0. Correct.
    
    # Vertex (0.5, h):
    # 1. 0.5 <= 1 (0.5)
    # 2. 0.5 + h*sqrt(3) = 0.5 + 1.5 = 2.0 ? No.
    # h = sqrt(3)/2. h*sqrt(3) = 3/2 = 1.5.
    # x + y*sqrt(3) = 0.5 + 1.5 = 2.0.
    # RHS is sqrt(3) approx 1.732.
    # So 2.0 <= 1.732 is FALSE.
    
    # My half plane equation was wrong.
    # Edge from (1,0) to (0.5, h).
    # Midpoint: (0.75, h/2).
    # Normal: (h, 0.5).
    # n.p = h*0.75 + 0.5*h/2 = 0.75h + 0.25h = h = sqrt(3)/2.
    # So equation: h*x + 0.5*y <= sqrt(3)/2.
    
    # Let's list the 6 inequalities for unit hexagon (side 1):
    # 1. x <= 1
    # 2. sqrt(3)/2 * x + 0.5 * y <= sqrt(3)/2
    # 3. -sqrt(3)/2 * x + 0.5 * y <= sqrt(3)/2
    # 4. -x <= 1  => x >= -1
    # 5. -sqrt(3)/2 * x - 0.5 * y <= sqrt(3)/2
    # 6. sqrt(3)/2 * x - 0.5 * y <= sqrt(3)/2
    
    # SDF is max of these 6 terms minus their respective RHS?
    # No, SDF for half plane n.p <= d is n.p - d.
    # So SDF(p) = max(
    #   x - 1,
    #   (sqrt(3)/2)*x + 0.5*y - sqrt(3)/2,
    #   (-sqrt(3)/2)*x + 0.5*y - sqrt(3)/2,
    #   -x - 1,
    #   (-sqrt(3)/2)*x - 0.5*y - sqrt(3)/2,
    #   (sqrt(3)/2)*x - 0.5*y - sqrt(3)/2
    # )
    
    # Let's test (0,0):
    # -1, -0.866, -0.866, -1, -0.866, -0.866. Max is -0.866.
    # Inradius is sqrt(3)/2 = 0.866. So SDF(0,0) = -0.866. Correct.
    
    # Let's test (1,0):
    # 0, 0.866-0.866=0, -0.866-0.866=-1.732, -2, ...
    # Max is 0. Correct.
    
    return np.max(np.array([
        x - 1.0,
        (math.sqrt(3)/2.0)*x + 0.5*y - math.sqrt(3)/2.0,
        (-math.sqrt(3)/2.0)*x + 0.5*y - math.sqrt(3)/2.0,
        -x - 1.0,
        (-math.sqrt(3)/2.0)*x - 0.5*y - math.sqrt(3)/2.0,
        (math.sqrt(3)/2.0)*x - 0.5*y - math.sqrt(3)/2.0
    ]))

def _hex_sdf_global(px, py, cx, cy, angle_deg):
    """Compute SDF of a point (px,py) relative to hexagon at (cx,cy) with rotation."""
    dx = px - cx
    dy = py - cy
    # Rotate point to local frame
    a = math.radians(-angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    return _hex_sdf_local(lx, ly)

def _objective_and_grad(x, outer_angle):
    """
    Objective: Outer Side Length + Penalty
    x: [x1, y1, a1, ..., x11, y11, a11]
    """
    positions = []
    for i in range(N):
        positions.append([x[i*3], x[i*3+1], x[i*3+2]])
    
    # 1. Compute Outer Side Length
    # The outer hexagon is centered at (0,0) with side L and angle outer_angle.
    # A point p is inside outer hexagon if SDF_outer(p) <= 0.
    # We want the minimum L such that all inner hexagons are inside.
    # This is equivalent to: L = max over all vertices of inner hexagons of (distance_to_boundary / cos(angle_diff))
    # Or simpler: 
    # For each inner hexagon, find the vertex that is furthest out in terms of "support".
    
    # Let's compute the required L by checking the support function of the union of inner hexagons
    # against the outer hexagon's support function.
    # Actually, simpler: 
    # The condition for a point p to be inside a regular hexagon of side L centered at origin with angle phi is:
    # max_k ( n_k . R(-phi) p ) <= L * sqrt(3) / 2
    # where n_k are the normals to the faces of the unit hexagon.
    
    # So, for each inner hexagon, we can compute the max support value in the outer frame.
    # But the inner hexagon rotates.
    
    # Let's just compute the vertices of the inner hexagons and check their distance to the outer boundary.
    # This is non-differentiable at the vertices.
    
    # To make it differentiable, we can use the SDF of the inner hexagons?
    # No, we need the outer size.
    
    # Alternative: 
    # Minimize L. 
    # Constraint: SDF_inner_i(p) <= 0 for all p in Outer? No.
    # Constraint: All points of Inner_i are in Outer.
    
    # Let's use the vertex method but smooth it.
    # The max distance of a vertex from the center determines L.
    # L = max_v ( r_v / cos(theta_v - theta_face_nearest) )
    
    # Let's stick to the previous `_min_enclosing_side` logic but make it part of the objective.
    # It is not differentiable.
    
    # How to make L differentiable?
    # L is the minimum side length. 
    # We can define the objective as:
    # f(x) = L(x) + Penalty(x, L(x))
    
    # Since L(x) is the max of a set of smooth functions (distance of each vertex),
    # it is convex and differentiable almost everywhere (subgradient exists).
    # We can use the subgradient or just the gradient of the active constraint.
    # L-BFGS-B uses gradients. If we provide the gradient of the active vertex, it might work.
    
    # Let's compute L and the gradient of L w.r.t positions.
    
    cx, cy = 0.0, 0.0
    oa_rad = math.radians(outer_angle)
    ca_o = math.cos(oa_rad)
    sa_o = math.sin(oa_rad)
    
    # Normals of outer hexagon faces (in global frame)
    # Face normals are at angles 30, 90, 150, 210, 270, 330 degrees relative to outer angle.
    # Wait, standard hexagon: vertices at 0, 60...
    # Face normals are perpendicular to edges.
    # Edge 0 (top right?): from (1,0) to (0.5, h). Normal (h, 0.5). Angle 30 deg.
    # So face normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    face_angles = [30, 90, 150, 210, 270, 330]
    normals = []
    for fa in face_angles:
        a = math.radians(fa + outer_angle)
        normals.append([math.cos(a), math.sin(a)])
    
    # For each inner hexagon, compute its vertices.
    # For each vertex, compute its projection onto each face normal.
    # The required "inradius" for that vertex is proj.
    # The required side L is 2 * max_proj / sqrt(3).
    
    max_proj = 0.0
    active_i = -1
    active_v = -1
    active_k = -1
    
    all_verts = []
    for i in range(N):
        ix, iy, iangle = positions[i]
        verts = []
        for k in range(6):
            a = math.radians(iangle + k * 60)
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)
            verts.append([vx, vy])
        all_verts.append(verts)
        
        for v_idx, v in enumerate(verts):
            for k_idx, n in enumerate(normals):
                proj = v[0]*n[0] + v[1]*n[1]
                if proj > max_proj:
                    max_proj = proj
                    active_i = i
                    active_v = v_idx
                    active_k = k_idx
    
    L = 2.0 * max_proj / math.sqrt(3)
    
    # Gradient of L
    # L = 2/sqrt(3) * max_proj
    # max_proj = v_active . n_active
    # v_active = pos_active + R(iangle_active) * e_k
    # e_k is the k-th vertex of unit hexagon.
    
    grad_L = np.zeros(3*N)
    
    if active_i >= 0:
        n_act = normals[active_k]
        i_act = active_i
        v_act = active_v
        a_act = positions[i_act][2]
        
        # v = [ix, iy] + [cos(a_act + v_act*60), sin(a_act + v_act*60)]
        # dv/dix = 1
        # dv/diy = 1
        # dv/da = [-sin(...), cos(...)]
        
        theta_v = math.radians(a_act + v_act * 60)
        e_x = math.cos(theta_v)
        e_y = math.sin(theta_v)
        
        # proj = (ix + e_x) * nx + (iy + e_y) * ny
        # dproj/dix = nx
        # dproj/diy = ny
        # dproj/da = (e_x * da/dix? No. e_x depends on da.
        # e_x = cos(a + phi). de_x/da = -sin(a+phi) = -e_y
        # e_y = sin(a + phi). de_y/da = cos(a+phi) = e_x
        # dproj/da = -e_y * nx + e_x * ny
        
        dL_dix = (2.0 / math.sqrt(3)) * n_act[0]
        dL_diy = (2.0 / math.sqrt(3)) * n_act[1]
        dL_da = (2.0 / math.sqrt(3)) * (-e_y * n_act[0] + e_x * n_act[1])
        
        idx = i_act * 3
        grad_L[idx] = dL_dix
        grad_L[idx+1] = dL_diy
        grad_L[idx+2] = dL_da
    
    # 2. Penalty for Overlaps
    # Use SDF of inner hexagons.
    # For each pair (i, j), compute the minimum distance between them.
    # We can approximate the distance by checking the SDF of j's center in i's frame? No.
    # We can check the SDF of i's vertices in j's frame and vice versa.
    # The minimum of these SDF values gives the gap.
    # If gap < 0, they overlap.
    
    penalty = 0.0
    grad_P = np.zeros(3*N)
    
    for i in range(N):
        for j in range(i+1, N):
            # Compute min SDF
            min_sdf = float('inf')
            active_v_i = -1
            active_v_j = -1
            
            # Check vertices of i in j
            for v_idx, v in enumerate(all_verts[i]):
                sdf = _hex_sdf_global(v[0], v[1], positions[j][0], positions[j][1], positions[j][2])
                if sdf < min_sdf:
                    min_sdf = sdf
                    active_v_i = v_idx
                    active_v_j = -1 # Marker for i-vertex
            
            # Check vertices of j in i
            for v_idx, v in enumerate(all_verts[j]):
                sdf = _hex_sdf_global(v[0], v[1], positions[i][0], positions[i][1], positions[i][2])
                if sdf < min_sdf:
                    min_sdf = sdf
                    active_v_i = -1 # Marker for j-vertex
                    active_v_j = v_idx
            
            if min_sdf < 0:
                # Overlap
                # Penalty = min_sdf^2
                # dP/dx = 2 * min_sdf * d(min_sdf)/dx
                # d(min_sdf)/dx is the gradient of the SDF at the active vertex.
                
                # We need the gradient of SDF at the active point.
                # SDF is max of 6 linear terms.
                # The gradient is the normal of the active face.
                
                if active_v_i >= 0:
                    # Active point is vertex v of hex i, inside hex j.
                    # SDF_j(v) = max_k ( n_k . (v - c_j) - d_k )
                    # We need to know which face of j is active.
                    # Let's re-compute to find the active face normal.
                    
                    vx, vy = all_verts[i][active_v_i]
                    jx, jy, ja = positions[j]
                    dx = vx - jx
                    dy = vy - jy
                    a_j = math.radians(-ja)
                    ca_j = math.cos(a_j)
                    sa_j = math.sin(a_j)
                    lx = dx * ca_j - dy * sa_j
                    ly = dx * sa_j + dy * ca_j
                    
                    # Find active face in local frame of j
                    h = math.sqrt(3)/2
                    terms = [
                        lx - 1.0,
                        h*lx + 0.5*ly - h,
                        -h*lx + 0.5*ly - h,
                        -lx - 1.0,
                        -h*lx - 0.5*ly - h,
                        h*lx - 0.5*ly - h
                    ]
                    max_term = max(terms)
                    k_act = terms.index(max_term)
                    
                    # Normal in local frame
                    if k_act == 0: n_loc = [1, 0]
                    elif k_act == 1: n_loc = [h, 0.5]
                    elif k_act == 2: n_loc = [-h, 0.5]
                    elif k_act == 3: n_loc = [-1, 0]
                    elif k_act == 4: n_loc = [-h, -0.5]
                    else: n_loc = [h, -0.5]
                    
                    # Rotate normal to global frame
                    a_j_pos = math.radians(ja)
                    ca_j_pos = math.cos(a_j_pos)
                    sa_j_pos = math.sin(a_j_pos)
                    nx_g = n_loc[0]*ca_j_pos - n_loc[1]*sa_j_pos
                    ny_g = n_loc[0]*sa_j_pos + n_loc[1]*ca_j_pos
                    
                    # Gradient of SDF_j(v) w.r.t v is (nx_g, ny_g)
                    # v = c_i + e_i
                    # dSDF/dc_i = (nx_g, ny_g)
                    # dSDF/de_i = (nx_g, ny_g) . d e_i / da_i
                    # e_i = (cos(theta), sin(theta))
                    # de_i/da = (-sin, cos)
                    
                    theta_i = math.radians(positions[i][2] + active_v_i * 60)
                    de_dx = -math.sin(theta_i)
                    de_dy = math.cos(theta_i)
                    
                    dSDF_dci = [nx_g, ny_g]
                    dSDF_dai = nx_g * de_dx + ny_g * de_dy
                    
                    # Gradient of SDF_j(v) w.r.t c_j is - (nx_g, ny_g)
                    # Gradient w.r.t a_j?
                    # SDF depends on c_j and a_j.
                    # dSDF/dc_j = -n_g
                    # dSDF/da_j = ?
                    # The normal n_g rotates with a_j.
                    # n_g = R(a_j) n_loc
                    # dn_g/da_j = R'(a_j) n_loc = R(a_j) J n_loc
                    # J = [[0, -1], [1, 0]]
                    # This is complex. Let's approximate or ignore rotation gradient for penalty?
                    # Or just use finite difference for rotation? No, L-BFGS needs analytic.
                    # Let's assume the penalty is dominated by position.
                    # We will add the position gradient.
                    
                    idx_i = i * 3
                    idx_j = j * 3
                    
                    # P = sdf^2
                    # dP = 2 * sdf * dSDF
                    coeff = 2.0 * min_sdf
                    
                    grad_P[idx_i] += coeff * dSDF_dci[0]
                    grad_P[idx_i+1] += coeff * dSDF_dci[1]
                    grad_P[idx_i+2] += coeff * dSDF_dai
                    
                    grad_P[idx_j] += coeff * (-dSDF_dci[0])
                    grad_P[idx_j+1] += coeff * (-dSDF_dci[1])
                    # grad_P[idx_j+2] += ... (ignored for simplicity/stability)
                    
                    penalty += min_sdf * min_sdf
    
    total_obj = L + penalty
    total_grad = grad_L + grad_P
    
    return total_obj, total_grad

def construct():
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0]
    
    for oa in outer_angles:
        if _remaining_time() < 30.0:
            break
            
        # Initial config: Lattice
        positions = []
        positions.append([0.0, 0.0, 0.0])
        dist = 2.0 * math.sqrt(3)
        for k in range(6):
            a = math.radians(60 * k)
            positions.append([dist * math.cos(a), dist * math.sin(a), 0.0])
        dist2 = dist * math.sqrt(3)
        for k in range(4):
            a = math.radians(30 + 60 * k)
            positions.append([dist2 * math.cos(a), dist2 * math.sin(a), 0.0])
        
        x0 = []
        for p in positions:
            x0.extend(p)
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                args=(oa,),
                method='L-BFGS-B',
                jac=True, # We return (f, g)
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            x_opt = res.x
            opt_positions = []
            for i in range(N):
                opt_positions.append([x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]])
            
            # Verify
            # Re-calculate side
            side, _ = _objective_and_grad(x_opt, oa)
            
            # Check validity with strict SAT
            if _check_validity(opt_positions, (0.0, 0.0), side, oa, tol=1e-5):
                if side < best_score:
                    best_score = side
                    best_result = (opt_positions, (0.0, 0.0), side, oa)
                    
        except Exception as e:
            pass

    if best_result is not None:
        return best_result
    
    # Fallback
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = math.radians(60 * k)
        positions.append([dist * math.cos(a), dist * math.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = math.radians(30 + 60 * k)
        positions.append([dist2 * math.cos(a), dist2 * math.sin(a), 0.0])
    
    side, _ = _objective_and_grad(np.array([p for pos in positions for p in pos]), 0.0)
    return positions, (0.0, 0.0), side, 0.0

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    cx, cy = outer_center
    
    # Check containment
    for (ix, iy, iangle) in inner_positions:
        for k in range(6):
            a = math.radians(iangle + k * 60)
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)
            
            # Check if (vx, vy) is in outer hex
            dx = vx - cx
            dy = vy - cy
            a_o = math.radians(-outer_angle)
            ca_o = math.cos(a_o)
            sa_o = math.sin(a_o)
            lx = dx * ca_o - dy * sa_o
            ly = dx * sa_o + dy * ca_o
            
            h = outer_side * math.sqrt(3) / 2.0
            # Check 6 inequalities
            if lx > outer_side + tol: return False
            if -lx > outer_side + tol: return False
            if h*lx + 0.5*ly > h + tol: return False
            if -h*lx + 0.5*ly > h + tol: return False
            if -h*lx - 0.5*ly > h + tol: return False
            if h*lx - 0.5*ly > h + tol: return False
            
    # Check overlaps
    all_verts = []
    for (ix, iy, iangle) in inner_positions:
        verts = []
        for k in range(6):
            a = math.radians(iangle + k * 60)
            verts.append([ix + math.cos(a), iy + math.sin(a)])
        all_verts.append(np.array(verts))
        
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False
