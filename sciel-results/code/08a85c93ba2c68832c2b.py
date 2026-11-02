import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
# Edge normals: for each edge from vert i to vert (i+1)%6
_edge_normals_raw = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _edge_normals_raw[_i] = [_nx / _norm, _ny / _norm]
_EDGE_NORMALS = _edge_normals_raw  # (6, 2) unit normals for edges of unit hex at angle 0

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    
    # Helper: Get vertices of all hexagons given positions and angles
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    def hexagons_disjoint_fast(h1, h2):
        """
        Fast SAT check for two unit hexagons.
        h1, h2: (x, y, angle_rad)
        Uses precomputed edge normals rotated by each hex's angle.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # Conservative early exit: max center distance for any contact is 2.0
        # (vertex-to-vertex of two unit hexagons). Use 4.5 as squared bound.
        if dist_sq > 4.5:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Compute vertices
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        
        # Rotate edge normals for each hex
        en = edge_normals  # (6, 2)
        en1x = en[:, 0] * c1 - en[:, 1] * s1
        en1y = en[:, 0] * s1 + en[:, 1] * c1
        en2x = en[:, 0] * c2 - en[:, 1] * s2
        en2y = en[:, 0] * s2 + en[:, 1] * c2
        
        # Check all 12 normals
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
                
        return False

    def compute_min_outer_side(positions, angles_deg):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        
        # Flatten to (72, 2)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        sqrt3 = math.sqrt(3)
        
        # Vectorized evaluation for a given phi
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            # 6 normal directions for the outer hexagon
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)  # (6,)
            ny = np.sin(normal_angles)  # (6,)
            
            # Project all vertices onto each normal: (72, 6)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)  # (72, 6)
            
            # For each direction, get max absolute projection
            max_projs = np.max(abs_proj, axis=0)  # (6,)
            side_vals = (2.0 / sqrt3) * max_projs  # (6,)
            return np.max(side_vals)
        
        # Coarse scan to find approximate minimum
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Refine with minimize_scalar (bounded)
        try:
            lo = max(0.0, best_phi - 2.0)
            hi = min(60.0, best_phi + 2.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        """Check if all hexagons are pairwise disjoint."""
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Initial Configuration: Honeycomb packing (optimal for hexagons)
    # In honeycomb, adjacent hexagons have centers at distance sqrt(3) (edge-to-edge)
    # or 2.0 (vertex-to-vertex). The honeycomb lattice vectors are:
    #   a1 = (3, 0)  -- but that's for the "flat" orientation
    # Actually for unit hexagons in honeycomb:
    #   Horizontal spacing: 3.0 (center to center, flat-to-flat... no)
    # Let me think: unit hexagon has circumradius 1, inradius sqrt(3)/2
    # In honeycomb packing, centers are at distance sqrt(3) apart (edge-to-edge touching)
    # The lattice: rows offset by sqrt(3)/2 in y, and 1.5 in x for alternating rows
    
    # Actually the standard honeycomb: place hexagons so they share edges.
    # Center-to-center distance = 2 * inradius = sqrt(3)
    # Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    # No wait. For regular hexagons of side 1:
    #   Width (flat-to-flat) = sqrt(3)
    #   Height (vertex-to-vertex) = 2
    # In honeycomb, neighbors are at distance sqrt(3) apart.
    # Lattice: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    # Generate 12 positions in a roughly circular honeycomb patch
    # Use coordinates (i, j) in lattice basis, centered
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    
    for idx, (i, j) in enumerate(lattice_coords):
        initial_positions[idx] = i * a1 + j * a2
        initial_angles[idx] = 0.0
            
    best_side = float('inf')
    best_positions = initial_positions.copy()
    best_angles = initial_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    current_positions = initial_positions.copy()
    current_angles = initial_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0005
    cooling = 0.9997
    step_size = 0.3
    accept_count = 0
    total_count = 0
    step_decay = 0.9999  # slowly reduce step size
    
    step_count = 0
    max_steps = 80000
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.8 and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 8.0) # degrees
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        # Check validity - only check against other hexagons
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx: continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
                
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.01: step_size = 0.01
            continue
            
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
                    
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.01: step_size = 0.01

    # Polish Step
    # Use Nelder-Mead on the best configuration found
    # Variables: 36 (x, y, angle for each hex)
    
    def compute_overlap_penalty(pos, angles_rad, soft=50.0):
        """Smooth surrogate for overlap: sum of soft-max of penetration depths."""
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Estimate overlap depth using center distance
                    dx = h1[0] - h2[0]
                    dy = h1[1] - h2[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Max possible center distance for non-overlap is ~2.0
                    # Use a soft penalty based on how deep the overlap is
                    overlap_depth = max(0.0, 2.0 - dist)
                    penalty += soft * overlap_depth * overlap_depth
        return penalty

    def polish_objective(params):
        # Unpack
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        angles_rad = np.radians(ang)
        
        # Check hard validity first
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Return large penalty + smooth surrogate to guide back
                    return 1e4 + compute_overlap_penalty(pos, angles_rad, soft=100.0)
        
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        return side

    def polish_objective_smooth(params):
        """Smooth objective using soft constraints for better gradient-like behavior."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        angles_rad = np.radians(ang)
        
        # Always compute side (may be invalid but gives a signal)
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        
        # Add smooth overlap penalty
        penalty = compute_overlap_penalty(pos, angles_rad, soft=50.0)
        
        return side + penalty

    # Initial params for polish
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = best_positions[i, 0]
        init_params[3*i+1] = best_positions[i, 1]
        init_params[3*i+2] = best_angles[i]
        
    # Run multiple polish passes with decreasing tolerances
    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 5.0:
        try:
            # First pass: smooth objective to find a good basin
            res = minimize(polish_objective_smooth, init_params, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            new_pos = np.zeros((N, 2))
            new_ang = np.zeros(N)
            for i in range(N):
                new_pos[i, 0] = res.x[3*i]
                new_pos[i, 1] = res.x[3*i+1]
                new_ang[i] = res.x[3*i+2]
            
            if is_valid_configuration(new_pos, new_ang):
                cand_side, cand_phi, cand_cx, cand_cy = compute_min_outer_side(new_pos, new_ang)
                if cand_side < best_side:
                    best_side = cand_side
                    best_positions = new_pos
                    best_angles = new_ang
                    best_phi = cand_phi
                    best_cx = cand_cx
                    best_cy = cand_cy
                    
                    # Second pass: hard constraint, tighter tolerance
                    if time.time() - t0 < DEADLINE - 3.0:
                        init_params2 = np.zeros(3*N)
                        for i in range(N):
                            init_params2[3*i] = best_positions[i, 0]
                            init_params2[3*i+1] = best_positions[i, 1]
                            init_params2[3*i+2] = best_angles[i]
                        
                        res2 = minimize(polish_objective, init_params2, method='Nelder-Mead', 
                                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
                        
                        if res2.fun < 1e3:  # Not penalized
                            new_pos2 = np.zeros((N, 2))
                            new_ang2 = np.zeros(N)
                            for i in range(N):
                                new_pos2[i, 0] = res2.x[3*i]
                                new_pos2[i, 1] = res2.x[3*i+1]
                                new_ang2[i] = res2.x[3*i+2]
                            
                            if is_valid_configuration(new_pos2, new_ang2):
                                cand_side2, cand_phi2, cand_cx2, cand_cy2 = compute_min_outer_side(new_pos2, new_ang2)
                                if cand_side2 < best_side:
                                    best_side = cand_side2
                                    best_positions = new_pos2
                                    best_angles = new_ang2
                                    best_phi = cand_phi2
                                    best_cx = cand_cx2
                                    best_cy = cand_cy2
                                    
                                    # Third pass: ultra-tight
                                    if time.time() - t0 < DEADLINE - 1.0:
                                        init_params3 = np.zeros(3*N)
                                        for i in range(N):
                                            init_params3[3*i] = best_positions[i, 0]
                                            init_params3[3*i+1] = best_positions[i, 1]
                                            init_params3[3*i+2] = best_angles[i]
                                        
                                        res3 = minimize(polish_objective, init_params3, method='Nelder-Mead', 
                                                       options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12})
                                        
                                        if res3.fun < 1e3:
                                            new_pos3 = np.zeros((N, 2))
                                            new_ang3 = np.zeros(N)
                                            for i in range(N):
                                                new_pos3[i, 0] = res3.x[3*i]
                                                new_pos3[i, 1] = res3.x[3*i+1]
                                                new_ang3[i] = res3.x[3*i+2]
                                            
                                            if is_valid_configuration(new_pos3, new_ang3):
                                                cand_side3, cand_phi3, cand_cx3, cand_cy3 = compute_min_outer_side(new_pos3, new_ang3)
                                                if cand_side3 < best_side:
                                                    best_side = cand_side3
                                                    best_positions = new_pos3
                                                    best_angles = new_ang3
                                                    best_phi = cand_phi3
                                                    best_cx = cand_cx3
                                                    best_cy = cand_cy3
        except Exception:
            pass

    # Final verification and result
    # Re-validate the best configuration
    final_valid = is_valid_configuration(best_positions, best_angles)
    if not final_valid:
        # Fallback to initial honeycomb if somehow invalid
        best_positions = initial_positions.copy()
        best_angles = initial_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Final precise computation of outer hexagon
    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Construct inner list: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
