"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True
        
    def check_all_overlaps(positions, angles):
        """Check if any pair of hexagons overlaps."""
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return True
        return False

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        positions: list of (x, y) for each inner hexagon
        angles: list of rotation angles for each inner hexagon
        outer_angle_deg: rotation of outer hexagon
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        # Get all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
            
        all_verts = np.vstack(all_verts)
        
        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        
        side = 2 * max_proj / math.sqrt(3)
        return side

    def get_config_score(positions, angles, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    # Generate honeycomb base positions (uncentered, in axial coords)
    def honeycomb_base_positions():
        """Generate 12 hexagons in a honeycomb pattern. Returns list of (x,y) in axial coords."""
        # Axial coordinate system for hex grid
        # Use a compact arrangement: 3-4-3-2 or similar
        # Let's use a 4x3 grid in axial coords
        positions = []
        # Row 0: 4 hexagons
        for c in range(4):
            positions.append((c * math.sqrt(3), 0.0))
        # Row 1: 3 hexagons (offset)
        for c in range(3):
            positions.append((c * math.sqrt(3) + math.sqrt(3)/2, 1.5))
        # Row 2: 4 hexagons
        for c in range(4):
            positions.append((c * math.sqrt(3), 3.0))
        # Row 3: 1 hexagon (to make 12 total: 4+3+4+1=12)
        positions.append((math.sqrt(3)/2, 4.5))
        return positions

    # Convert axial positions to Cartesian with scale and rotation
    def transform_positions(base_positions, scale, rotation_deg, offsets):
        """
        base_positions: list of (x,y) in base honeycomb coords
        scale: uniform scaling factor
        rotation_deg: global rotation in degrees
        offsets: list of (dx, dy) per hexagon (local offsets)
        Returns list of (x, y) final positions
        """
        rot_rad = math.radians(rotation_deg)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        result = []
        for i, (bx, by) in enumerate(base_positions):
            # Scale
            sx = bx * scale
            sy = by * scale
            # Rotate
            rx = sx * cos_r - sy * sin_r
            ry = sx * sin_r + sy * cos_r
            # Add offset
            fx = rx + offsets[i][0]
            fy = ry + offsets[i][1]
            result.append((fx, fy))
        return result

    # Objective function for scipy optimizer
    def objective(params, base_positions, outer_angle):
        """
        params: [scale, rotation_deg, dx0, dy0, dx1, dy1, ..., dx11, dy11]
        Returns the outer side length or a penalty if overlapping.
        """
        if time.time() > deadline:
            return 1e10
            
        scale = params[0]
        rotation_deg = params[1]
        offsets = [(params[2+2*i], params[3+2*i]) for i in range(N)]
        
        positions = transform_positions(base_positions, scale, rotation_deg, offsets)
        angles = [0.0] * N  # Keep all inner hexagons aligned for now
        
        if check_all_overlaps(positions, angles):
            return 1e10
            
        return compute_min_side(positions, angles, outer_angle)

    # Try multiple outer angles
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    base_pos = honeycomb_base_positions()
    
    # Initial guess: scale=1, rotation=0, no offsets
    # Bounds for differential evolution
    bounds = [(0.8, 1.5)]  # scale
    bounds.append((-30, 30))  # rotation_deg
    for i in range(N):
        bounds.append((-0.5, 0.5))  # dx
        bounds.append((-0.5, 0.5))  # dy

    x0 = [1.0, 0.0] + [0.0] * (2*N)
    
    # Try different outer angles
    outer_angles_to_try = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
    
    for oa in outer_angles_to_try:
        if time.time() > deadline - 20:
            break
        
        # Use Nelder-Mead for refinement
        try:
            result = minimize(
                objective,
                x0,
                args=(base_pos, oa),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if result.fun < best_score and result.fun < 1e9:
                scale = result.x[0]
                rot = result.x[1]
                offsets = [(result.x[2+2*i], result.x[3+2*i]) for i in range(N)]
                positions = transform_positions(base_pos, scale, rot, offsets)
                
                # Center the configuration
                cx = sum(p[0] for p in positions) / N
                cy = sum(p[1] for p in positions) / N
                positions = [(x - cx, y - cy) for x, y in positions]
                
                angles = [0.0] * N
                score = compute_min_side(positions, angles, oa)
                
                if score < best_score and not check_all_overlaps(positions, angles):
                    best_score = score
                    best_positions = positions
                    best_angles = angles
                    best_outer_angle = oa
        except Exception:
            pass

    # If Nelder-Mead didn't find a good solution, try differential evolution
    if best_score > 10 or best_positions is None:
        for oa in [0, 30, 60]:
            if time.time() > deadline - 30:
                break
            try:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    args=(base_pos, oa),
                    maxiter=100,
                    tol=1e-4,
                    seed=42,
                    polish=False
                )
                
                if de_result.fun < best_score and de_result.fun < 1e9:
                    scale = de_result.x[0]
                    rot = de_result.x[1]
                    offsets = [(de_result.x[2+2*i], de_result.x[3+2*i]) for i in range(N)]
                    positions = transform_positions(base_pos, scale, rot, offsets)
                    
                    # Center the configuration
                    cx = sum(p[0] for p in positions) / N
                    cy = sum(p[1] for p in positions) / N
                    positions = [(x - cx, y - cy) for x, y in positions]
                    
                    angles = [0.0] * N
                    score = compute_min_side(positions, angles, oa)
                    
                    if score < best_score and not check_all_overlaps(positions, angles):
                        best_score = score
                        best_positions = positions
                        best_angles = angles
                        best_outer_angle = oa
            except Exception:
                pass

    # Fallback: If no valid configuration was found, use a sparse grid
    if best_positions is None or best_score == float('inf'):
        positions = []
        angles = []
        for i in range(N):
            r = i // 4
            c = i % 4
            positions.append((c * 4.0, r * 4.0))
            angles.append(0.0)
        best_positions = positions
        best_angles = angles
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, angles, 0.0)

    # Final cleanup: ensure we return the best valid configuration
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0,0)] * N
        final_angles = [0] * N
        final_outer = 0
        best_score = 10.0 # Large penalty

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
