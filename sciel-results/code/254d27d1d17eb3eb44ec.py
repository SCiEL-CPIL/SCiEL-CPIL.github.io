"""
Strategy: Global Lattice Search + Local Refinement.
1. Define a hexagonal lattice grid for relative positions.
2. Systematically search for a set of 11 non-overlapping points on this grid
   that minimizes the bounding hexagon size.
3. Use the best discrete configuration as a seed for a fast local refinement
   (Nelder-Mead) to fine-tune positions and rotations.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    SQRT3 = math.sqrt(3)
    
    # Pre-compute constants for speed
    def get_hex_vertices(cx, cy, angle_deg):
        # Vertices of a regular hexagon with side 1 (radius 1)
        # Note: Side length 1 implies circumradius 1.
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(angle_deg):
        # Normals to the sides for Separating Axis Theorem
        axes = []
        for i in range(6):
            # Side normals are at 30, 90, 150... degrees relative to vertex 0
            # If vertices are at 0, 60..., sides are perpendicular to radii at 30, 90...
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def is_disjoint_fast(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        # The outer hexagon must contain all inner hexagons.
        # For a regular hexagon container with angle theta, the width in direction n is 2*R*cos(angle_diff)
        # Actually, the distance from center to side (apothem) is R * sqrt(3)/2.
        # If we project all points onto the 6 normal directions of the outer hexagon,
        # the maximum projection P determines the required apothem A = P.
        # Side length S = 2 * A / sqrt(3).
        
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))

        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj

        return 2.0 * max_proj / SQRT3

    def evaluate_config(params, num_outer_samples=12):
        """
        params: [x0, y0, a0, x1, y1, a1, ...]
        Returns: outer_side_length + penalty_for_overlap
        """
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[i*3], params[i*3+1]))
            angles.append(params[i*3+2])

        # Center of mass to origin for fair comparison
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]

        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))

        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint_fast(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    # Penalty based on penetration depth approximation
                    dx = shifted_centers[i][0] - shifted_centers[j][0]
                    dy = shifted_centers[i][1] - shifted_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance for disjoint hexagons (side 1) is sqrt(3) if aligned, 
                    # but can be less if rotated. Use a conservative penalty.
                    penetration = max(0.0, SQRT3 - dist)
                    overlap_penalty += 1000.0 * penetration * penetration + 10.0

        if overlap_penalty > 0:
            return overlap_penalty + 100.0 # Hard fail for optimization

        best_side = float('inf')
        for k in range(num_outer_samples):
            oa_deg = k * 60.0 / num_outer_samples
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side

        return best_side

    # --- Phase 1: Discrete Lattice Search ---
    # We search for relative positions on a hex grid.
    # Spacing 1.0 is tight. Let's use spacing 0.9 to allow some "breathing room" 
    # but keep it discrete. Or better, just use integer axial coordinates 
    # which correspond to distance sqrt(3) * spacing.
    # If we use spacing=1.0 in axial coords, distance between neighbors is sqrt(3) ~ 1.732.
    # This is exactly the distance for touching hexagons.
    
    # Generate candidate axial coordinates within a radius
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # List of axial coordinates within a certain "hex distance" from origin
    # We need 11 points. A cluster of 11 hexagons fits in a hexagon of side 3?
    # Side 3 hexagon has 19 cells. Side 2 has 7.
    # So we are looking for a subset of 11 from a larger grid, or a specific shape.
    
    # Let's generate all points within axial distance 3 from origin.
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Axial distance formula: (abs(q) + abs(r) + abs(q+r)) / 2
            dist = (abs(q) + abs(r) + abs(q+r)) // 2
            if dist <= 3:
                candidates.append((q, r))
                
    # We need to choose 11 points from candidates such that no two are adjacent (distance 1)?
    # No, they just need to be disjoint. In a hex grid, disjoint means distance >= 1 (touching) or > 1.
    # If we place centers on the grid points of the *dual* lattice (centers of hexagons),
    # two hexagons are disjoint if their centers are at least sqrt(3) apart (for aligned hexes).
    # In axial coordinates, distance 1 corresponds to Cartesian distance sqrt(3).
    # So we need to select 11 points from the axial grid such that the axial distance between any pair is >= 1.
    # Wait, axial distance 1 means they are neighbors. Neighbors touch.
    # So we can select any 11 points, as long as we don't select the *same* point.
    # Actually, if we select points at axial distance 1, the hexagons touch. That is allowed (disjoint interiors).
    # So the problem reduces to: Select 11 points from the infinite hex grid to minimize the bounding hexagon.
    # Since the grid is discrete, we can enumerate subsets? 11 from ~37 points is C(37,11) which is huge.
    
    # Better: Use a heuristic search on the discrete grid.
    # Start with a known good configuration (e.g., a 3x3 block minus corners, or a cross).
    # Let's try to find a "compact" set of 11 grid points.
    
    # Pre-defined good discrete patterns (axial coords)
    # Pattern 1: 3x3 block (9 points) + 2 extensions
    # Center (0,0). Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # This is the 7-point flower. Add 4 more.
    
    discrete_configs = []
    
    # Config A: The 7-point flower + 4 points at distance 2
    # Flower: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add: (2,0), (0,2), (-2,2), (-2,0) ... need to check overlaps.
    # Let's just use a few hand-picked discrete sets that are known to be valid packings.
    
    # Set 1: Hexagon of side 2 (7 pts) + 4 pts attached to flat sides
    # Side 2 hexagon vertices in axial: (0,0), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0) -> wait, that's 6.
    # Standard hex grid coords for side 2 (radius 1 in axial):
    # q from -1 to 1, r from -1 to 1, with constraint |q+r|<=1? No.
    # Let's just generate a dense cluster and let the optimizer fix it.
    
    # Initial discrete seeds:
    # 1. 3x3 square in axial (which is a rhombus in cartesian)
    # 2. Cross shape
    # 3. Random valid discrete sets
    
    def generate_random_discrete_set():
        # Pick 11 points from a 5x5 axial area
        pts = set()
        while len(pts) < 11:
            q = random.randint(-2, 2)
            r = random.randint(-2, 2)
            pts.add((q, r))
        return list(pts)
    
    # We will run a few random discrete searches, then refine the best one.
    
    best_discrete_params = None
    best_discrete_score = float('inf')
    
    # Run 50 random discrete configurations
    for _ in range(50):
        if time.time() > deadline - 60.0:
            break
            
        pts = generate_random_discrete_set()
        # Convert to cartesian
        params = []
        for q, r in pts:
            x, y = axial_to_cartesian(q, r, 1.0)
            params.extend([x, y, 0.0])
            
        score = evaluate_config(params, num_outer_samples=6)
        if score < best_discrete_score:
            best_discrete_score = score
            best_discrete_params = params.copy()
            
    # Also try the "flower + extensions" manually
    flower = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Add 4 more points to minimize bounding box
    # Try adding (2,0), (2,-1), (0,2), (-2,2)
    ext1 = flower + [(2,0), (2,-1), (0,2), (-2,2)]
    ext2 = flower + [(2,0), (1,1), (-2,2), (-1,-1)] # Check overlaps
    ext3 = flower + [(2,-1), (1,1), (-2,1), (-1,-2)]
    
    for ext in [ext1, ext2, ext3]:
        # Check validity of discrete set (no duplicate points)
        if len(set(ext)) != 11:
            continue
        params = []
        for q, r in ext:
            x, y = axial_to_cartesian(q, r, 1.0)
            params.extend([x, y, 0.0])
        score = evaluate_config(params, num_outer_samples=6)
        if score < best_discrete_score:
            best_discrete_score = score
            best_discrete_params = params.copy()
            
    if best_discrete_params is None:
        # Fallback: just use the first random one
        pts = generate_random_discrete_set()
        best_discrete_params = []
        for q, r in pts:
            x, y = axial_to_cartesian(q, r, 1.0)
            best_discrete_params.extend([x, y, 0.0])

    # --- Phase 2: Local Refinement ---
    # Use Nelder-Mead to refine positions and angles.
    # We allow angles to vary.
    
    current_params = best_discrete_params
    current_score = evaluate_config(current_params, num_outer_samples=12)
    
    # Define objective for optimizer
    def objective(p):
        return evaluate_config(p, num_outer_samples=8)
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, current_params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if res.fun < current_score:
            current_params = res.x
            current_score = res.fun
    except Exception:
        pass
    
    # Second polish with higher resolution outer angle check
    if time.time() < deadline - 10.0:
        def objective_high_res(p):
            return evaluate_config(p, num_outer_samples=24)
            
        try:
            res2 = minimize(objective_high_res, current_params, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res2.fun < current_score:
                current_params = res2.x
                current_score = res2.fun
        except Exception:
            pass

    # --- Finalize ---
    centers = []
    angles = []
    for i in range(N):
        centers.append((current_params[i*3], current_params[i*3+1]))
        angles.append(current_params[i*3+2])

    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]

    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))

    # Validate
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint_fast(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to a known valid configuration if optimization failed
        # Use the initial discrete params which were likely valid
        params_fb = best_discrete_params
        centers_fb = []
        angles_fb = []
        for i in range(N):
            centers_fb.append((params_fb[i*3], params_fb[i*3+1]))
            angles_fb.append(params_fb[i*3+2])
        cx_avg_fb = sum(c[0] for c in centers_fb) / N
        cy_avg_fb = sum(c[1] for c in centers_fb) / N
        shifted_centers_fb = [(x - cx_avg_fb, y - cy_avg_fb) for x, y in centers_fb]
        
        hexagons_fb = []
        hex_axes_fb = []
        for i in range(N):
            cx, cy = shifted_centers_fb[i]
            a = angles_fb[i]
            hexagons_fb.append(get_hex_vertices(cx, cy, a))
            hex_axes_fb.append(get_hex_axes(a))
            
        # Check if fallback is valid
        valid_fb = True
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint_fast(hexagons_fb[i], hexagons_fb[j], hex_axes_fb[i], hex_axes_fb[j]):
                    valid_fb = False
                    break
            if not valid_fb:
                break
        
        if valid_fb:
            centers = centers_fb
            angles = angles_fb
            shifted_centers = shifted_centers_fb
            hexagons = hexagons_fb
            hex_axes = hex_axes_fb
            valid = True
        else:
            # Last resort: Grid placement
            # Place 11 hexagons in a 4x3 grid (with one missing)
            # Axial: q in 0..3, r in 0..2
            coords = []
            for q in range(4):
                for r in range(3):
                    coords.append((q, r))
            # Take first 11
            coords = coords[:11]
            centers = []
            angles = [0.0]*N
            for q, r in coords:
                x, y = axial_to_cartesian(q, r, 1.0)
                centers.append((x, y))
            cx_avg = sum(c[0] for c in centers) / N
            cy_avg = sum(c[1] for c in centers) / N
            shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
            hexagons = [get_hex_vertices(c[0], c[1], a) for c, a in zip(shifted_centers, angles)]
            hex_axes = [get_hex_axes(a) for a in angles]
            valid = True

    # Calculate final side length
    best_side = float('inf')
    best_oa = 0.0
    # Fine scan for outer angle
    for oa_deg in range(0, 600, 2): # 2 deg steps
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    # Even finer scan around best_oa
    for oa_deg in range(int(best_oa*10)-10, int(best_oa*10)+10):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa

    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])

    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
