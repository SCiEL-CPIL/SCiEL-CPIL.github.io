"""
Caging formulation: Optimize inner hexagon positions via Simulated Annealing 
to minimize the side length of the minimal containing regular hexagon.
The outer hexagon is optimized analytically for each inner configuration 
by scanning its rotation angle to minimize the maximum extent of the 
inner convex hull along the outer hexagon's face normals.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

# Precompute constants for hexagon geometry
S3 = math.sqrt(3)
COS60 = 0.5
SIN60 = S3 / 2.0
PI = math.pi
TWO_PI = 2 * PI

def get_hex_vertices_fast(cx, cy, angle_deg):
    """Fast computation of hexagon vertices using precomputed trig values."""
    a = math.radians(angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    
    # Base vertices at angles 0, 60, 120, 180, 240, 300
    # (1, 0), (0.5, S3/2), (-0.5, S3/2), (-1, 0), (-0.5, -S3/2), (0.5, -S3/2)
    # Rotated by a:
    # x' = x*ca - y*sa
    # y' = x*sa + y*ca
    
    v = np.empty((6, 2))
    # i=0: (1, 0) -> (ca, sa)
    v[0, 0] = cx + ca
    v[0, 1] = cy + sa
    # i=1: (0.5, S3/2) -> (0.5*ca - S3/2*sa, 0.5*sa + S3/2*ca)
    v[1, 0] = cx + 0.5*ca - SIN60*sa
    v[1, 1] = cy + 0.5*sa + SIN60*ca
    # i=2: (-0.5, S3/2) -> (-0.5*ca - S3/2*sa, -0.5*sa + S3/2*ca)
    v[2, 0] = cx - 0.5*ca - SIN60*sa
    v[2, 1] = cy - 0.5*sa + SIN60*ca
    # i=3: (-1, 0) -> (-ca, -sa)
    v[3, 0] = cx - ca
    v[3, 1] = cy - sa
    # i=4: (-0.5, -S3/2) -> (-0.5*ca + S3/2*sa, -0.5*sa - S3/2*ca)
    v[4, 0] = cx - 0.5*ca + SIN60*sa
    v[4, 1] = cy - 0.5*sa - SIN60*ca
    # i=5: (0.5, -S3/2) -> (0.5*ca + S3/2*sa, 0.5*sa - S3/2*ca)
    v[5, 0] = cx + 0.5*ca + SIN60*sa
    v[5, 1] = cy + 0.5*sa - SIN60*ca
    
    return v

def construct():
    t0 = time.time()
    
    # Helper: Get vertices of a regular hexagon centered at (cx, cy) with side 1 and angle deg
    def get_hex_vertices(cx, cy, angle_deg):
        return get_hex_vertices_fast(cx, cy, angle_deg)

    # Helper: Check if two hexagons are disjoint
    # Fast rejection: Bounding box check
    # Hexagon side 1: Width = 2 (if flat top) or sqrt(3) (if pointy top). 
    # Actually, width depends on rotation. 
    # Max extent in any direction is 1 (radius). 
    # So if distance between centers > 2, they are disjoint.
    # If distance < sqrt(3) ~ 1.732, they might overlap? 
    # Min distance between centers for disjoint hexagons (touching edges) is sqrt(3).
    # So if dist < sqrt(3) - epsilon, they definitely overlap.
    # We will use a SAT check if dist < 2.0.
    
    def hexagons_disjoint(h1, h2):
        # h1, h2 are (x, y, angle)
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint
        if dist_sq > 4.0: # dist > 2.0
            return True
            
        # If very close, likely overlap, but let's check precisely
        # Get vertices
        v1 = get_hex_vertices(h1[0], h1[1], h1[2])
        v2 = get_hex_vertices(h2[0], h2[1], h2[2])
        
        # SAT: Check 6 normals from each hexagon
        # Normals for hexagon 1
        for i in range(6):
            # Edge vector
            ex = v1[(i+1)%6, 0] - v1[i, 0]
            ey = v1[(i+1)%6, 1] - v1[i, 1]
            # Normal
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            # Project v1 and v2 onto normal
            proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
            proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
            proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
            proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
                
        for i in range(6):
            ex = v2[(i+1)%6, 0] - v2[i, 0]
            ey = v2[(i+1)%6, 1] - v2[i, 1]
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
            proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
            proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
            proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
                
        return False

    def is_valid_configuration(inner):
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    return False
        return True

    def compute_min_outer_side(inner):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        We optimize the rotation angle of the outer hexagon.
        The outer hexagon is centered at the centroid of the convex hull of the inner configuration.
        """
        # 1. Get all vertices of inner hexagons
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        # 2. Compute centroid of the convex hull (more robust center)
        # Use the centroid of the vertices of the convex hull for better symmetry
        try:
            hull = ConvexHull(all_verts)
            hull_verts = all_verts[hull.vertices]
            center = np.mean(hull_verts, axis=0)
        except Exception:
            center = np.mean(all_verts, axis=0)
            
        offsets = all_verts - center
        
        # 3. Find optimal rotation angle phi for outer hexagon
        # Outer hexagon normals are at angles phi + 30 + k*60
        # Required side S = max_k ( 2/sqrt(3) * max_v | n_k . offset_v | )
        # We scan phi in [0, 60)
        
        sqrt3 = S3
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = math.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + PI/6 + k * PI/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                # Project all offsets
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                # S = 2/sqrt(3) * max_proj
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Fine scan around best_phi
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.1):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = math.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + PI/6 + k * PI/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg

        # Polish: Use scipy minimize on the phi angle to refine
        # The objective function for a given phi is the max extent
        def obj_phi(phi_rad):
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi_rad + PI/6 + k * PI/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            return max_extent

        # Start from best_phi found in scan
        phi_init = math.radians(best_phi)
        # Try a few local optimizations from nearby starts to avoid local minima
        for start_offset in [0, 0.1, -0.1, 0.2, -0.2]:
            start_phi = phi_init + start_offset
            try:
                res = minimize(obj_phi, start_phi, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = math.degrees(res.x[0]) % 60.0
            except:
                pass

        return best_side, best_phi, center

    # Initial Configuration: Honeycomb Cluster
    # A dense packing of 12 hexagons.
    # We use a 4x3 grid of hexagons in a honeycomb pattern.
    # Rows: 3, Cols: 4
    # Hexagons in row r, col c:
    # x = c * 1.5 + (r % 2) * 0.75  (if flat-top, horizontal distance is 1.5*side? No, side=1, width=2, dist=1.5 for offset rows? 
    # Let's use standard honeycomb coordinates.
    # Flat-top hexagons:
    # x_dist = 1.5 * side = 1.5
    # y_dist = sqrt(3) * side = sqrt(3)
    # Offset x for odd rows: 0.75? No, 1.5/2 = 0.75.
    # Let's check:
    # Hex 0 at (0,0). Vertices at (1,0), (0.5, S3/2), ...
    # Hex 1 at (1.5, S3/2). Vertices at (2.5, S3/2), (2, S3), ...
    # Distance between (0.5, S3/2) and (1.5, S3/2) is 1.0. They share an edge?
    # Edge of Hex 0: (1,0) to (0.5, S3/2). Midpoint (0.75, S3/4).
    # Edge of Hex 1: (1.5, S3/2) to (1.0, S3). Midpoint (1.25, 3S3/4).
    # Wait, Hex 1 vertices:
    # (1.5+1, S3/2) = (2.5, S3/2)
    # (1.5+0.5, S3/2+S3/2) = (2, S3)
    # (1.5-0.5, S3/2+S3/2) = (1, S3)
    # (1.5-1, S3/2) = (0.5, S3/2)
    # (1.5-0.5, S3/2-S3/2) = (1, 0)
    # (1.5+0.5, S3/2-S3/2) = (2, 0)
    
    # Hex 0 vertices:
    # (1,0), (0.5, S3/2), (-0.5, S3/2), (-1, 0), (-0.5, -S3/2), (0.5, -S3/2)
    
    # Hex 0 right vertex: (1,0). Hex 1 left-bottom vertex: (1,0). They touch at a point?
    # Hex 0 edge: (1,0)-(0.5, S3/2).
    # Hex 1 edge: (1,0)-(0.5, S3/2).
    # Yes, they share an edge. So distance (1.5, S3/2) is correct for honeycomb.
    
    # Let's generate 12 hexagons in a 4x3 block.
    # Rows 0, 1, 2. Cols 0, 1, 2, 3.
    initial_inner = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5 + (r % 2) * 0.75
            y = r * S3
            initial_inner.append([x, y, 0.0])
            
    # Center the configuration
    cx = np.mean([h[0] for h in initial_inner])
    cy = np.mean([h[1] for h in initial_inner])
    for h in initial_inner:
        h[0] -= cx
        h[1] -= cy

    best_inner = [list(h) for h in initial_inner]
    best_side, best_phi, best_center = compute_min_outer_side(best_inner)
    
    current_inner = [list(h) for h in initial_inner]
    current_side, _, _ = compute_min_outer_side(current_inner)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    step_size = 0.5
    
    step_count = 0
    max_steps = 20000 # Hard limit
    
    while time.time() - t0 < DEADLINE and step_count < max_steps:
        step_count += 1
        
        # Perturb one random hexagon
        idx = random.randint(0, N - 1)
        new_inner = [list(h) for h in current_inner]
        
        # Perturb position and angle
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0) # degrees
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        # Check validity (disjointness)
        valid = True
        # Only need to check the moved hexagon against others
        for j in range(N):
            if j == idx: continue
            if not hexagons_disjoint(new_inner[idx], new_inner[j]):
                valid = False
                break
                
        if not valid:
            # Reject
            T *= cooling
            if T < T_min: T = T_min
            continue
            
        # Compute new side length
        new_side, new_phi, new_center = compute_min_outer_side(new_inner)
        
        # Accept if better or with probability exp(-delta/T)
        if new_side < current_side:
            current_inner = new_inner
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_inner = [list(h) for h in current_inner]
                best_phi = new_phi
                best_center = new_center
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_inner = new_inner
                current_side = new_side
                if new_side < best_side:
                    best_side = new_side
                    best_inner = [list(h) for h in current_inner]
                    best_phi = new_phi
                    best_center = new_center
                    
        T *= cooling
        if T < T_min: T = T_min

    # Final Polish: Optimize the positions of all hexagons simultaneously using Nelder-Mead
    # with a smooth penalty for overlap and containment.
    # This can refine the SA result significantly.
    
    def flatten_inner(inner):
        return np.array([item for sublist in inner for item in sublist])

    def unflatten_inner(flat):
        inner = []
        for i in range(N):
            inner.append([flat[3*i], flat[3*i+1], flat[3*i+2]])
        return inner

    def objective(flat):
        inner = unflatten_inner(flat)
        
        # 1. Check containment (compute min outer side)
        # We want to minimize the side length.
        # But we also need to ensure disjointness.
        # If we just minimize side length, they might overlap.
        # So we add a penalty for overlap.
        
        # Compute min outer side
        try:
            side, phi, center = compute_min_outer_side(inner)
        except:
            return 1e10

        # Compute overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    # Calculate penetration depth?
                    # Approximate by distance between centers if they overlap
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Min dist for disjoint is sqrt(3) ~ 1.732
                    # If dist < sqrt(3), they overlap.
                    # Penalty proportional to (sqrt(3) - dist)^2
                    if dist < S3:
                        penalty += (S3 - dist)**2
        
        return side + 100.0 * penalty

    # Start from best_inner found by SA
    current_flat = flatten_inner(best_inner)
    
    # Time check
    time_left = DEADLINE - (time.time() - t0)
    if time_left > 10.0: # Need at least 10s for polish
        try:
            # Run Nelder-Mead
            res = minimize(objective, current_flat, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 200, 'maxfev': 500})
            
            if res.fun < best_side: # Check if we actually improved the side length (penalty should be 0 if valid)
                # Verify validity
                polished_inner = unflatten_inner(res.x)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not hexagons_disjoint(polished_inner[i], polished_inner[j]):
                            valid = False
                            break
                        if j == i: continue # Redundant but safe
                if valid:
                    best_side, best_phi, best_center = compute_min_outer_side(polished_inner)
                    best_inner = polished_inner
        except:
            pass

    # Final Result
    # Ensure best_inner is a list of lists [x, y, angle]
    # Center is tuple (x, y)
    # Side is float
    # Angle is float (outer rotation)
    
    return best_inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_phi)

# EVOLVE-BLOCK-END
