# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a smooth optimization approach (L-BFGS-B) to minimize the outer hexagon side length,
    treating collision and containment constraints as soft penalties in the objective function.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    sqrt3 = math.sqrt(3)
    N = 12

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def compute_objective(params):
        # params structure:
        # 0..23: 12 centers (x, y)
        # 24..35: 12 angles
        # 26: outer center x
        # 27: outer center y
        # 28: outer angle
        # 29: outer side length s

        centers = params[:24].reshape(12, 2)
        angles = params[24:36]
        outer_cx = params[26]
        outer_cy = params[27]
        outer_angle = params[28]
        s = params[29]
        
        # Ensure s is positive
        if s <= 0:
            s = 1e-6
        s = max(s, 1e-6)

        # 1. Compute all inner hexagon vertices
        inner_verts_list = []
        for i in range(12):
            verts = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            inner_verts_list.append(verts)

        # 2. Compute outer hexagon vertices
        outer_verts = get_hex_vertices(outer_cx, outer_cy, s, outer_angle)
        
        # Outer edge normals (outward)
        # For a CCW polygon, the outward normal to edge (v_i, v_{i+1}) is (dy, -dx) normalized?
        # Let's check. Edge vector e = v_{i+1} - v_i.
        # Normal n = (-e_y, e_x) is left normal (inward for CCW).
        # So outward normal is (e_y, -e_x).
        # Let's verify with a simple case.
        # Hexagon center (0,0), side 1, angle 0.
        # v0 = (1, 0), v1 = (0.5, sqrt(3)/2).
        # e = (-0.5, sqrt(3)/2).
        # Outward normal should be roughly (0.5, -sqrt(3)/2) direction? No, that's inward.
        # The edge is at the "top right". The outward normal points away from center.
        # Center is (0,0). Midpoint of v0-v1 is (0.75, sqrt(3)/4).
        # Vector from center to midpoint is (0.75, sqrt(3)/4).
        # Let's check dot product with (e_y, -e_x) = (sqrt(3)/2, 0.5).
        # Dot = 0.75*sqrt(3)/2 + sqrt(3)/4 * 0.5 = sqrt(3)/2 * (0.75 + 0.25) = sqrt(3)/2 > 0.
        # So (e_y, -e_x) is indeed the outward normal direction.
        
        outer_normals = []
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1)%6]
            e = p2 - p1
            n = np.array([e[1], -e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                n = n / norm
            outer_normals.append(n)
        
        # 3. Objective: Minimize s + penalties
        
        # Penalty 1: Containment
        # All inner vertices must be inside the outer hexagon.
        # Condition: For all inner vertices v, and all outer edges (p1, p2),
        # cross(p2-p1, v-p1) >= -eps.
        # This is equivalent to: dot(outward_normal, v - p1) <= eps (if normal is outward).
        # Let's re-derive.
        # cross(e, v-p1) >= 0 means v is to the left of e (for CCW).
        # If e is CCW, left is inward. So cross >= 0 means v is inside.
        # Wait, the prompt says: "cross(p2 - p1, v - p1) must be >= -1e-6".
        # This implies the vertices are ordered such that the interior is on the "positive cross" side.
        # My get_hex_vertices generates CCW vertices.
        # So for CCW, interior is to the left.
        # cross(e, v-p1) > 0 means v is to the left of e.
        # So the condition is satisfied if cross >= -1e-6.
        
        # We want to penalize violations where cross < 0.
        # Let violation = max(0, -cross).
        # We sum violations over all inner vertices and all outer edges.
        
        containment_penalty = 0.0
        for i in range(12):
            verts = inner_verts_list[i]
            for v in verts:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1)%6]
                    e = p2 - p1
                    vec = v - p1
                    cross = e[0]*vec[1] - e[1]*vec[0]
                    if cross < 0:
                        containment_penalty += (-cross) ** 2  # Smooth penalty
        
        # Penalty 2: Separation (SAT)
        # For each pair of inner hexagons, we need a separating axis.
        # If they are not separated on any axis, they collide.
        # We want to penalize the "minimum separation" over all axes.
        # If min_sep > 0, they are separated.
        # If min_sep < 0, they overlap.
        # We want to penalize max(0, -min_sep).
        
        separation_penalty = 0.0
        for i in range(12):
            verts1 = inner_verts_list[i]
            for j in range(i+1, 12):
                verts2 = inner_verts_list[j]
                
                # Collect axes (normals) from both hexagons
                axes = []
                # Normals from hex1
                for k in range(6):
                    p1 = verts1[k]
                    p2 = verts1[(k+1)%6]
                    e = p2 - p1
                    n = np.array([e[1], -e[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9:
                        n = n / norm
                    axes.append(n)
                # Normals from hex2
                for k in range(6):
                    p1 = verts2[k]
                    p2 = verts2[(k+1)%6]
                    e = p2 - p1
                    n = np.array([e[1], -e[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9:
                        n = n / norm
                    axes.append(n)
                
                min_sep = float('inf')
                for axis in axes:
                    # Project verts1
                    proj1 = verts1 @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    # Project verts2
                    proj2 = verts2 @ axis
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Separation distance
                    # If max1 < min2, sep = min2 - max1
                    # If max2 < min1, sep = min1 - max2
                    # Else sep = min(max1 - min2, max2 - min1) (negative if overlap)
                    
                    sep1 = min2 - max1
                    sep2 = min1 - max2
                    
                    if sep1 > 0 or sep2 > 0:
                        # They are separated on this axis
                        sep = max(sep1, sep2)
                    else:
                        # They overlap on this axis
                        sep = min(sep1, sep2)
                    
                    if sep < min_sep:
                        min_sep = sep
                
                # If min_sep < 0, they are colliding.
                # We want to penalize the depth of collision.
                # Use a smooth penalty like max(0, -min_sep)^2
                if min_sep < 0:
                    separation_penalty += (-min_sep) ** 2

        # Objective: Minimize s + penalties
        # We want s to be small, but penalties to be zero.
        # If penalties are non-zero, the configuration is invalid.
        # We use a large weight for penalties to ensure they are minimized first.
        weight_cont = 1000.0
        weight_sep = 1000.0
        
        obj = s + weight_cont * containment_penalty + weight_sep * separation_penalty
        
        return obj

    # Initial guess
    # Hexagonal lattice arrangement
    # 1 center, 6 ring, 5 more?
    # Let's use a 3-4-3-2 or similar grid.
    # Or just random.
    
    best_result = None
    best_obj = float('inf')
    
    # Multiple random restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial params
        # Centers in a rough hexagonal pattern
        centers = []
        # 1 center
        centers.append([0, 0])
        # 6 around
        for i in range(6):
            ang = i * math.pi / 3
            centers.append([2.5 * math.cos(ang), 2.5 * math.sin(ang)])
        # 5 more
        for i in range(5):
            ang = i * math.pi / 3 + math.pi/6
            centers.append([5.0 * math.cos(ang), 5.0 * math.sin(ang)])
            
        # Add some randomness
        centers = np.array(centers)
        centers += np.random.normal(0, 0.5, centers.shape)
        
        angles = np.zeros(12)
        angles += np.random.normal(0, 0.1, 12)
        
        outer_cx = 0
        outer_cy = 0
        outer_angle = 0
        s = 10.0
        
        params = np.concatenate([centers.flatten(), angles, [outer_cx, outer_cy, outer_angle, s]])
        
        # Optimize
        try:
            res = minimize(compute_objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
        except Exception:
            continue
            
        if res.success and res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x

    # Extract best result
    if best_result is None:
        # Fallback
        return np.zeros((12,3)), np.array([0,0,0]), 10.0

    centers = best_result[:24].reshape(12, 2)
    angles = best_result[24:36]
    outer_cx = best_result[26]
    outer_cy = best_result[27]
    outer_angle = best_result[28]
    s = best_result[29]
    
    # Construct return arrays
    inner_ret = np.zeros((12, 3))
    for i in range(12):
        inner_ret[i, 0] = centers[i, 0]
        inner_ret[i, 1] = centers[i, 1]
        inner_ret[i, 2] = angles[i]
        
    outer_ret = np.array([outer_cx, outer_cy, outer_angle])
    
    return inner_ret, outer_ret, s

# EVOLVE-BLOCK-END
