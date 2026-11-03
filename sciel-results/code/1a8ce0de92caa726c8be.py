import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        base = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
        verts = base @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def hex_edges(verts):
        """Return edge vectors and outward normals for a convex hexagon (CCW)."""
        edges = np.roll(verts, -1, axis=0) - verts
        # Outward normals: rotate edge by -90 deg (for CCW polygon)
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms
        return edges, normals

    def sat_min_separation(verts_a, verts_b):
        """
        Using SAT, find the minimum translation needed to separate two convex hexagons.
        Returns (min_overlap, best_axis) where min_overlap is the maximum penetration depth
        along the best separating axis (negative means separated).
        """
        edges_a, normals_a = hex_edges(verts_a)
        edges_b, normals_b = hex_edges(verts_b)
        
        all_normals = np.vstack([normals_a, normals_b])
        
        min_overlap = float('inf')
        best_axis = None
        
        for n in all_normals:
            # Project both polygons onto this axis
            proj_a = verts_a @ n
            proj_b = verts_b @ n
            
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            
            # Overlap on this axis
            overlap = min(max_a, max_b) - max(min_a, min_b)
            
            if overlap < 0:
                # Separated along this axis
                return 0.0, n
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = n
        
        if best_axis is None:
            return 0.0, np.array([1.0, 0.0])
        
        # Determine direction: push A away from B
        center_a = np.mean(verts_a, axis=0)
        center_b = np.mean(verts_b, axis=0)
        diff = center_a - center_b
        if np.dot(diff, best_axis) < 0:
            best_axis = -best_axis
        
        return min_overlap, best_axis

    def calc_outer_side(positions, angles, outer_angle=0.0):
        """Minimum side of regular hexagon (centered at origin, given rotation) containing all inner hexagons."""
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def overlap_constraint(params, i, j):
        """
        SLSQP constraint: return the minimum separation distance between hex i and hex j.
        Must be >= 0 for non-overlap.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        verts_i = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts_j = get_hex_verts(positions[j, 0], positions[j, 1], 1.0, angles[j])
        
        sep, axis = sat_min_separation(verts_i, verts_j)
        # sep > 0 means overlap by 'sep' along axis; we need separation >= 0
        # The constraint value is the "gap": if overlapping, gap is negative
        # We want gap >= 0, so constraint = -sep (since sep is penetration depth)
        # Actually: if sep > 0, they overlap by sep. Gap = -sep.
        # If sep == 0, they are touching or separated. Gap = 0.
        return -sep  # We need this >= 0

    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        return calc_outer_side(positions, angles, 0.0)

    # --- Step 1: Constructive initial placement on hexagonal lattice ---
    # For flat-top hexagons (angle=0), the lattice vectors are:
    # a1 = (3, 0)  [distance between centers of adjacent hexagons sharing a vertical edge]
    # a2 = (1.5, 3*sqrt(3)/2)  [distance for the other set of neighbors]
    # Actually for unit hexagons:
    # The distance between centers of two adjacent hexagons (sharing an edge) is 2*apothem = sqrt(3)
    # For pointy-top (vertex at 0 deg): 
    #   Adjacent centers are at distance sqrt(3) in 6 directions.
    #   Directions: 0, 60, 120, 180, 240, 300 degrees
    #   But the actual lattice: for hexagons of side 1, center-to-center distance = sqrt(3)
    #   Lattice vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    #   Wait, let me recalculate. For a regular hexagon of side 1:
    #   apothem = sqrt(3)/2
    #   Center-to-center for edge-sharing = 2 * apothem = sqrt(3)
    #   The 6 neighbor directions from center: at angles 0, 60, 120, 180, 240, 300
    #   But these are the directions to the midpoints of edges.
    #   So neighbors are at: sqrt(3) * (cos(k*60), sin(k*60)) for k=0..5
    
    sqrt3 = math.sqrt(3)
    # Lattice vectors for hexagonal tiling of unit hexagons:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    # Place 12 hexagons in a compact arrangement on the lattice
    # Use a 4x3 rectangular patch of the hexagonal lattice (slightly rotated for compactness)
    # Or a hexagonal arrangement: center + 6 neighbors + 5 more
    
    # Strategy: use a hexagonal cluster shape
    # Ring 0: 1 hex (center)
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes (we only need 5 more)
    # Total for rings 0-1: 7, need 5 more from ring 2
    
    # Actually, let's try several topologies and pick the best:
    
    def lattice_positions(ring_structure):
        """Generate positions on hexagonal lattice given ring structure."""
        positions = []
        # Ring 0: center
        positions.append(np.array([0.0, 0.0]))
        
        # Ring 1: 6 neighbors
        for k in range(6):
            ang = k * math.pi / 3
            pos = sqrt3 * np.array([math.cos(ang), math.sin(ang)])
            positions.append(pos)
        
        # Ring 2: 12 neighbors (at distance 2*sqrt(3) in 6 dirs, or sqrt(3) in 6 offset dirs)
        # Ring 2 positions: v1+v2, v1-v2, 2*v1, 2*v2, -v1+v2, -v1-v2, -2*v1, -2*v2, etc.
        ring2 = []
        # The 12 positions at "distance 2" in hex lattice:
        ring2_offsets = [
            2*v1, 2*v2, 2*(v1-v2),
            -2*v1, -2*v2, -2*(v1-v2),
            v1+v2, v1-v2, -v1+v2, -v1-v2,
            2*v1-v2, 2*v2-v1
        ]
        # Actually let me be more careful. In a hex lattice with basis v1, v2:
        # Ring 1: ±v1, ±v2, ±(v1-v2)  -- 6 positions
        # Ring 2: ±2v1, ±2v2, ±2(v1-v2), ±(v1+v2), ±(2v1-v2), ±(2v2-v1) -- but that's 12
        # Let me just enumerate all lattice points at hex-distance 2:
        # (i, j) in lattice coords where |i|+|j|+|i-j| = 2 (hex distance)
        # i, j range: -2 to 2
        # hex_dist(i,j) = max(|i|, |j|, |i-j|) ... no
        # Actually hex distance = (|i| + |j| + |i+j|) / 2 for axial coords
        # Let me use cube coords: (x, y, z) with x+y+z=0
        # hex_dist = max(|x|, |y|, |z|)
        # v1 corresponds to (1, 0, -1), v2 corresponds to (0, 1, -1)
        # So position = i*v1 + j*v2 = (i, j, -(i+j))
        # hex_dist = max(|i|, |j|, |i+j|)
        
        ring2_pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                d = max(abs(i), abs(j), abs(i+j))
                if d == 2:
                    pos = i * v1 + j * v2
                    ring2_pts.append(pos)
        
        # We need 5 more from ring 2 (total 1 + 6 + 5 = 12)
        # Pick 5 that form a compact shape - take consecutive ones
        # Sort by angle for a compact arc
        ring2_sorted = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
        # Take 5 consecutive (they'll form an arc)
        for k in range(5):
            positions.append(ring2_sorted[k])
        
        return np.array(positions)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal cluster (1 + 6 + 5 from ring 2)
    pos1 = lattice_positions(None)
    initial_configs.append(pos1)
    
    # Config 2: 3x4 rectangular grid on hex lattice
    pos2 = []
    for row in range(3):
        for col in range(4):
            # Offset every other row
            offset = 0.5 * v1 if row % 2 == 1 else 0
            pos = col * v1 + row * v2 + offset
            pos2.append(pos)
    initial_configs.append(np.array(pos2))
    
    # Config 3: 4x3 rectangular grid
    pos3 = []
    for row in range(4):
        for col in range(3):
            offset = 0.5 * v1 if row % 2 == 1 else 0
            pos = col * v1 + row * v2 + offset
            pos3.append(pos)
    initial_configs.append(np.array(pos3))
    
    # Config 4: Compact hexagonal (1+6+5) but different 5 from ring 2
    ring2_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d = max(abs(i), abs(j), abs(i+j))
            if d == 2:
                pos = i * v1 + j * v2
                ring2_pts.append(pos)
    ring2_sorted2 = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
    pos4 = [np.array([0.0, 0.0])]
    for k in range(6):
        ang = k * math.pi / 3
        pos4.append(sqrt3 * np.array([math.cos(ang), math.sin(ang)]))
    # Take 5 from different part of ring 2
    for k in range(5, 10):
        pos4.append(ring2_sorted2[k % len(ring2_sorted2)])
    initial_configs.append(np.array(pos4))
    
    # All use angle = 0 (pointy-top, matching the lattice)
    best_result = None
    best_score = float('inf')
    
    # --- Step 2: SLSQP optimization with SAT constraints for each initial config ---
    for cfg_idx, init_pos in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Center the configuration
        center = np.mean(init_pos, axis=0)
        init_pos = init_pos - center
        
        # Initial params: positions (24) + angles (12, all 0)
        x0 = np.concatenate([init_pos.flatten(), np.zeros(N)])
        
        # Define constraints for all pairs
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda params, i=i, j=j: overlap_constraint(params, i, j)
                })
        
        # Bounds: keep positions in a reasonable range
        bounds = []
        for i in range(N):
            bounds.append((-6, 6))  # x
            bounds.append((-6, 6))  # y
            bounds.append((0, 0))   # angle fixed at 0 (lattice-compatible)
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                # Verify the solution
                final_pos = res.x[:2*N].reshape(N, 2)
                final_ang = res.x[2*N:]
                
                # Check all pairs are truly non-overlapping
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        vi = get_hex_verts(final_pos[i,0], final_pos[i,1], 1.0, final_ang[i])
                        vj = get_hex_verts(final_pos[j,0], final_pos[j,1], 1.0, final_ang[j])
                        sep, _ = sat_min_separation(vi, vj)
                        if sep > 1e-6:  # Overlapping
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side = calc_outer_side(final_pos, final_ang, 0.0)
                    if side < best_score:
                        best_score = side
                        best_result = res.x.copy()
                        
        except Exception as e:
            pass
    
    # --- Step 3: If SLSQP didn't work well, try with slightly perturbed angles ---
    if best_result is None or best_score > 10:
        # Try with small angle perturbations
        for cfg_idx, init_pos in enumerate(initial_configs[:2]):
            if time.time() > deadline - 15:
                break
            
            center = np.mean(init_pos, axis=0)
            init_pos = init_pos - center
            
            # Allow small angle variations
            angles0 = np.random.uniform(-5, 5, N)
            x0 = np.concatenate([init_pos.flatten(), angles0])
            
            constraints = []
            for i in range(N):
                for j in range(i+1, N):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda params, i=i, j=j: overlap_constraint(params, i, j)
                    })
            
            bounds = []
            for i in range(N):
                bounds.append((-6, 6))
                bounds.append((-6, 6))
                bounds.append((-30, 30))
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                
                if res.fun < best_score:
                    final_pos = res.x[:2*N].reshape(N, 2)
                    final_ang = res.x[2*N:]
                    
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            vi = get_hex_verts(final_pos[i,0], final_pos[i,1], 1.0, final_ang[i])
                            vj = get_hex_verts(final_pos[j,0], final_pos[j,1], 1.0, final_ang[j])
                            sep, _ = sat_min_separation(vi, vj)
                            if sep > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_score = res.fun
                        best_result = res.x.copy()
                        
            except Exception:
                pass
    
    # --- Step 4: Final polish with Nelder-Mead on best result ---
    if best_result is not None and time.time() < deadline - 10:
        try:
            # Use a simpler objective with penalty for overlap
            def penalized_objective(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side = calc_outer_side(positions, angles, 0.0)
                
                # Check overlaps
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        vi = get_hex_verts(positions[i,0], positions[i,1], 1.0, angles[i])
                        vj = get_hex_verts(positions[j,0], positions[j,1], 1.0, angles[j])
                        sep, _ = sat_min_separation(vi, vj)
                        if sep > 0:
                            penalty += sep ** 2
                
                return side + 100 * penalty
            
            res_polish = minimize(
                penalized_objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res_polish.fun < best_score + 1e-6:
                final_pos = res_polish.x[:2*N].reshape(N, 2)
                final_ang = res_polish.x[2*N:]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        vi = get_hex_verts(final_pos[i,0], final_pos[i,1], 1.0, final_ang[i])
                        vj = get_hex_verts(final_pos[j,0], final_pos[j,1], 1.0, final_ang[j])
                        sep, _ = sat_min_separation(vi, vj)
                        if sep > 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side = calc_outer_side(final_pos, final_ang, 0.0)
                    if side < best_score:
                        best_score = side
                        best_result = res_polish.x.copy()
                        
        except Exception:
            pass
    
    # --- Fallback: honeycomb lattice arrangement ---
    if best_result is None or best_score > 10:
        sqrt3 = math.sqrt(3)
        # Standard honeycomb: 12 hexagons in a compact arrangement
        # Using the lattice: center + 6 + 5
        v1 = np.array([sqrt3, 0.0])
        v2 = np.array([sqrt3/2.0, 1.5])
        
        coords = [np.array([0.0, 0.0])]
        for k in range(6):
            ang = k * math.pi / 3
            coords.append(sqrt3 * np.array([math.cos(ang), math.sin(ang)]))
        
        # Ring 2: pick 5
        ring2_pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                d = max(abs(i), abs(j), abs(i+j))
                if d == 2:
                    pos = i * v1 + j * v2
                    ring2_pts.append(pos)
        ring2_sorted = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
        for k in range(5):
            coords.append(ring2_sorted[k])
        
        coords = np.array(coords)
        center = np.mean(coords, axis=0)
        coords = coords - center
        
        angles = np.zeros(N)
        best_score = calc_outer_side(coords, angles, 0.0)
        best_result = np.concatenate([coords.flatten(), angles])
    
    # Extract final result
    positions = best_result[:2*N].reshape(N, 2)
    angles = best_result[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
    
    final_side = calc_outer_side(positions, angles, 0.0)
    
    return inner, (0.0, 0.0), float(final_side), 0.0
