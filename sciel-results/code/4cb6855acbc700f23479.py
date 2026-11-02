"""Rewritten program for packing 12 unit hexagons in a regular hexagon.
Uses Hard Constraint Simulated Annealing with vectorized validity checks.
"""
import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices (side 1, angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # For a regular hexagon with side 1, circumradius is 1.
    # Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    angles = np.radians(np.arange(0, 360, 60))
    unit_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normal vectors for the 6 sides of a unit hexagon (outward pointing)
    # Side 1 (top right edge? No, let's define normals for the 6 half-planes)
    # A regular hexagon centered at origin with flat top/bottom? Or pointy?
    # Standard math: vertices at 0, 60...
    # Normals are perpendicular to edges.
    # Edge between v0 and v1. Midpoint angle 30 deg. Normal at 30 deg.
    # Normals at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.radians(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    # Apothem of unit hexagon
    apothem = SQRT3 / 2.0

    def get_hexagon_params(params):
        """Extract inner positions/angles and outer params from flat array."""
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        oc = (params[3*N], params[3*N+1])
        os = params[3*N+2]
        oa = params[3*N+3]
        return inner, oc, os, oa

    def is_valid(params):
        """
        Check if configuration is valid:
        1. All inner hexagons are inside the outer hexagon.
        2. All inner hexagons are disjoint.
        Returns True if valid, False otherwise.
        """
        inner, oc, os, oa = get_hexagon_params(params)
        ox, oy = oc
        outer_side = os
        outer_angle = oa
        
        if outer_side < 1.0: # Minimum side for 1 hex
             return False

        # 1. Containment Check
        # Transform outer hexagon to local frame? Or transform inner to outer frame?
        # Let's transform inner vertices to outer hexagon's local frame.
        # Outer hexagon centered at (ox, oy) with angle oa.
        # Local frame of outer hexagon:
        # Rotate by -oa, translate by -oc.
        # In this local frame, the outer hexagon is centered at (0,0), angle 0, side os.
        # The condition for a point (x,y) to be inside a regular hexagon (side s, angle 0) is:
        # |n_k . p| <= s * apothem for all 6 normals n_k.
        # Wait, the normals are fixed for the hexagon shape.
        # The hexagon is defined by 6 half-planes: n_k . p <= s * apothem.
        
        outer_apothem = outer_side * SQRT3 / 2.0
        
        # Precompute rotation matrix for outer hexagon
        cos_oa = math.cos(-math.radians(outer_angle))
        sin_oa = math.sin(-math.radians(outer_angle))
        
        # Check each inner hexagon
        for i in range(N):
            cx, cy, angle_deg = inner[i]
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Transform unit vertices to global coordinates
            # v_global = R(angle) @ unit_verts + center
            # Then transform to outer local frame:
            # v_local = R(-outer_angle) @ (v_global - outer_center)
            
            # v_global_x = cos_a * ux - sin_a * uy + cx
            # v_global_y = sin_a * ux + cos_a * uy + cy
            
            # v_local_x = cos_oa * (v_global_x - ox) - sin_oa * (v_global_y - oy)
            # v_local_y = sin_oa * (v_global_x - ox) + cos_oa * (v_global_y - oy)
            
            # We can combine the rotations.
            # Total rotation is angle - outer_angle.
            # Total translation is center - outer_center.
            
            rel_cx = cx - ox
            rel_cy = cy - oy
            
            # Rotation by (angle_deg - outer_angle)
            rel_angle_rad = angle_rad - math.radians(outer_angle)
            cos_ra = math.cos(rel_angle_rad)
            sin_ra = math.sin(rel_angle_rad)
            
            # Local vertices of inner hexagon in outer's frame
            # x_local = cos_ra * ux - sin_ra * uy + rel_cx
            # y_local = sin_ra * ux + cos_ra * uy + rel_cy
            
            # Compute for all 6 vertices at once
            # unit_verts is 6x2
            # We want to apply rotation and translation
            
            # Using matrix ops for speed
            # R = [[cos_ra, -sin_ra], [sin_ra, cos_ra]]
            # v_local = R @ unit_verts.T + [rel_cx, rel_cy]
            
            # Let's just do it in a loop or vectorized for the 6 vertices
            # Since N is small (12) and 6 verts, it's 72 points total.
            
            # Vectorized:
            # ux, uy = unit_verts.T
            # lx = cos_ra * ux - sin_ra * uy + rel_cx
            # ly = sin_ra * ux + cos_ra * uy + rel_cy
            
            lx = cos_ra * unit_verts[0] - sin_ra * unit_verts[1] + rel_cx
            ly = sin_ra * unit_verts[0] + cos_ra * unit_verts[1] + rel_cy
            
            # Check containment: max_k (n_k . p) <= outer_apothem
            # n_k are fixed (normals array 6x2)
            # p are (lx, ly)
            # proj = n_k . p = nx * lx + ny * ly
            # We need proj <= outer_apothem for all k.
            
            # proj is 6x6 matrix (6 normals, 6 vertices)
            # proj[i, j] = normals[i,0]*lx[j] + normals[i,1]*ly[j]
            # We want max(proj) <= outer_apothem
            
            # Actually, we can just check if any proj > outer_apothem
            # proj = normals @ np.column_stack([lx, ly]).T ? No.
            # normals is 6x2. lx, ly are 6x1 (or 1x6).
            # Let's make lx, ly 1D arrays of length 6.
            
            # proj = np.dot(normals, np.column_stack((lx, ly)).T) -> 6x6
            # This is overkill.
            
            # Simpler:
            # for each vertex j:
            #   for each normal k:
            #     if nx_k*lx_j + ny_k*ly_j > outer_apothem: return False
            
            # Vectorized for all vertices j:
            # proj_matrix = normals @ np.vstack([lx, ly]).T  # 6x6
            # if np.max(proj_matrix) > outer_apothem + 1e-9: return False
            
            proj_matrix = normals @ np.vstack([lx, ly]).T
            if np.max(proj_matrix) > outer_apothem + 1e-9:
                return False

        # 2. Separation Check (SAT)
        # For each pair (i, j)
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint(inner[i], inner[j]):
                    return False
                    
        return True

    def sat_disjoint(h1, h2):
        """Check if two hexagons are disjoint using SAT."""
        c1x, c1y, a1_deg = h1
        c2x, c2y, a2_deg = h2
        
        a1 = math.radians(a1_deg)
        a2 = math.radians(a2_deg)
        
        # Vertices of h1
        v1 = np.column_stack([
            c1x + np.cos(a1) * unit_verts[0] - np.sin(a1) * unit_verts[1],
            c1y + np.sin(a1) * unit_verts[0] + np.cos(a1) * unit_verts[1]
        ])
        
        # Vertices of h2
        v2 = np.column_stack([
            c2x + np.cos(a2) * unit_verts[0] - np.sin(a2) * unit_verts[1],
            c2y + np.sin(a2) * unit_verts[0] + np.cos(a2) * unit_verts[1]
        ])
        
        # SAT Axes: Normals of all edges of both hexagons
        # We can precompute the local normals and rotate them.
        # Local normals are `normals` (6x2).
        # Rotated normals for h1: R(a1) @ normals.T -> 2x6 -> T -> 6x2
        # Rotated normals for h2: R(a2) @ normals.T -> 2x6 -> T -> 6x2
        
        # Rotation matrix for a1
        R1 = np.array([[math.cos(a1), -math.sin(a1)], [math.sin(a1), math.cos(a1)]])
        R2 = np.array([[math.cos(a2), -math.sin(a2)], [math.sin(a2), math.cos(a2)]])
        
        axes1 = (R1 @ normals.T).T # 6x2
        axes2 = (R2 @ normals.T).T # 6x2
        
        all_axes = np.vstack([axes1, axes2])
        
        for axis in all_axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True # Disjoint
        
        return False # Overlapping

    def objective(params):
        """
        Objective function for SA.
        Returns outer side length if valid, +inf if invalid.
        """
        inner, oc, os, oa = get_hexagon_params(params)
        if not is_valid(params):
            return float('inf')
        return os

    def random_valid_config(rng):
        """Generate a random valid configuration."""
        # Try random positions
        for _ in range(100):
            params = np.zeros(3*N + 4)
            # Random positions in a box
            params[:3*N:3] = rng.uniform(-5, 5, N)
            params[1:3*N:3] = rng.uniform(-5, 5, N)
            params[2:3*N:3] = rng.uniform(0, 60, N)
            # Outer center 0,0, side 10, angle 0
            params[3*N] = 0.0
            params[3*N+1] = 0.0
            params[3*N+2] = 10.0
            params[3*N+3] = 0.0
            
            if is_valid(params):
                return params
        # Fallback: Honeycomb
        return honeycomb_config()

    def honeycomb_config():
        """Generate a known valid honeycomb configuration."""
        params = np.zeros(3*N + 4)
        # 3 rows, 4 cols
        idx = 0
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * 3.0
                y = (r - 1) * 2.598
                if r % 2 == 1:
                    x += 1.5
                params[3*idx] = x
                params[3*idx+1] = y
                params[3*idx+2] = 0.0
                idx += 1
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 8.0
        params[3*N+3] = 0.0
        return params

    # Main Loop: Simulated Annealing with Hard Constraints
    best_params = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Initial Configurations
    initial_configs = [honeycomb_config()]
    # Add some random ones if time permits
    for _ in range(3):
        initial_configs.append(random_valid_config(rng))

    for init_params in initial_configs:
        if time.time() > deadline - 20:
            break
            
        current = init_params.copy()
        current_score = objective(current)
        
        if current_score < best_score:
            best_score = current_score
            best_params = current.copy()
            
        if time.time() > deadline - 15:
            break

        # SA Parameters
        t0 = 1.0
        t_end = 0.01
        max_steps = 2000
        alpha = 0.99 # Cooling factor
        
        temperature = t0
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Generate neighbor
            neighbor = current.copy()
            # Perturb
            # Move 1-2 inner hexagons
            num_moves = rng.integers(1, 3)
            for _ in range(num_moves):
                i = rng.integers(0, N)
                # Random delta
                dx = rng.normal(0, 0.5)
                dy = rng.normal(0, 0.5)
                da = rng.normal(0, 10)
                
                neighbor[3*i] += dx
                neighbor[3*i+1] += dy
                neighbor[3*i+2] += da
                
            # Occasionally move outer
            if rng.random() < 0.1:
                neighbor[3*N+2] += rng.normal(0, 0.1) # Side
                # Ensure side is positive
                if neighbor[3*N+2] < 1.0:
                    neighbor[3*N+2] = 1.0

            neighbor_score = objective(neighbor)
            
            # Acceptance
            if neighbor_score < current_score:
                current = neighbor
                current_score = neighbor_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current.copy()
            else:
                # Metropolis criterion
                # If neighbor is invalid (inf), reject.
                # If valid but worse, accept with prob exp(-delta/T)
                if neighbor_score == float('inf'):
                    # Reject invalid moves in hard constraint SA?
                    # Yes, usually.
                    pass
                else:
                    delta = neighbor_score - current_score
                    if delta > 0:
                        prob = math.exp(-delta / temperature)
                        if rng.random() < prob:
                            current = neighbor
                            current_score = neighbor_score
                    else:
                        # Should not happen if we checked <
                        pass
            
            temperature *= alpha
            if temperature < t_end:
                break

    # Final Refinement with Nelder-Mead on a smooth objective?
    # Or just return best_params.
    # Let's try a quick local refinement if time permits.
    if time.time() < deadline - 10 and best_params is not None:
        # Use scipy minimize on a smooth penalty function derived from distances
        # But we need to define it.
        # For now, just return the best found.
        pass

    inner, oc, os, oa = get_hexagon_params(best_params)
    
    # Final validity check and expansion if needed
    if not is_valid(best_params):
        # Expand outer side
        for _ in range(100):
            best_params[3*N+2] += 0.1
            if is_valid(best_params):
                break
        inner, oc, os, oa = get_hexagon_params(best_params)

    return inner, oc, os, oa
