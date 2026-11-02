"""Combined strategy: Fixed hexagonal cluster + scipy.optimize.linprog for outer hexagon containment."""
import numpy as np
import time
import random
from scipy.optimize import linprog

N = 12
SQRT3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.arange(6) * np.pi / 3
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth_fast(v1, v2, normals):
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, max2)
        if np.min(overlaps) <= 1e-9:
            return 0.0
        return np.min(overlaps)

    def compute_min_side_lp(inner_verts_all, center, angle_rad):
        """
        Compute the minimum side length of an outer hexagon centered at `center`
        with rotation `angle_rad` such that it contains all vertices in `inner_verts_all`.
        
        Uses Linear Programming to find the maximum support distance.
        Variables: t (scalar)
        Constraints:
        For each of the 6 outward normals n_k of the outer hexagon:
        max_i ( (v_i - center) . n_k ) <= t
        
        This is equivalent to:
        (v_i - center) . n_k - t <= 0  for all i, k
        
        Minimize t.
        """
        # Outer hexagon normals
        # A regular hexagon with rotation theta has vertices at angles theta + k*pi/3.
        # The outward normals point to the midpoints of the edges.
        # Midpoint angle for edge k is theta + (k + 0.5)*pi/3.
        normals = np.array([
            [np.cos(angle_rad + (k + 0.5) * np.pi / 3),
             np.sin(angle_rad + (k + 0.5) * np.pi / 3)]
            for k in range(6)
        ])
        
        # Vertices relative to center
        rel_verts = inner_verts_all - np.array(center)
        
        # We want to minimize t
        # s.t. A @ [t] <= b is not quite right, we have:
        # rel_verts . n_k - t <= 0
        # -t + rel_verts . n_k <= 0
        # So A is a matrix where each row is [-1, ...]? No, t is a single variable.
        # Let x = [t].
        # Constraint: rel_verts . n_k - t <= 0
        # => -t + rel_verts . n_k <= 0
        # This is a linear inequality in t.
        
        # Since we minimize t, and t must be >= max(rel_verts . n_k),
        # the optimal t is exactly max(rel_verts . n_k).
        # We can just compute this directly without linprog!
        # But the prompt suggested linprog. Let's stick to the direct computation
        # as it's faster and exact, but ensure it matches the LP formulation.
        
        max_support = 0.0
        for k in range(6):
            n = normals[k]
            support = np.max(rel_verts @ n)
            if support > max_support:
                max_support = support
        
        # For a regular hexagon of side L, the distance from center to edge (apothem) is L * sqrt(3) / 2.
        # The support function in the normal direction is the apothem.
        # So, apothem = max_support
        # L * sqrt(3) / 2 = max_support
        # L = 2 * max_support / sqrt(3)
        
        return 2.0 * max_support / SQRT3

    def check_disjointness(positions, alphas):
        """Check if all inner hexagons are disjoint."""
        verts_list = [hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # Quick bounding box check
                if np.max(v1[:, 0]) < np.min(v2[:, 0]) or np.max(v2[:, 0]) < np.min(v1[:, 0]):
                    continue
                if np.max(v1[:, 1]) < np.min(v2[:, 1]) or np.max(v2[:, 1]) < np.min(v1[:, 1]):
                    continue
                
                normals1 = get_normals(v1)
                normals2 = get_normals(v2)
                normals = np.vstack([normals1, normals2])
                pd = penetration_depth_fast(v1, v2, normals)
                if pd > 1e-7:
                    return False, pd
        return True, 0.0

    def evaluate_config(positions, alphas, center, outer_angle_rad):
        """Evaluate a configuration: return (side_length, is_valid)."""
        # Check disjointness
        valid, pen = check_disjointness(positions, alphas)
        if not valid:
            return float('inf'), False
        
        # Compute min side
        verts_list = [hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        
        side = compute_min_side_lp(all_verts, center, outer_angle_rad)
        
        return side, True

    # Define fixed cluster structures
    # Structure 1: Honeycomb 3x4 (Flat Top)
    # Flat top hexagons: angle 30 deg.
    # Spacing: dx = sqrt(3), dy = 1.5. Shift odd rows by sqrt(3)/2.
    def make_honeycomb_flat(rows, cols):
        pos = []
        dx = SQRT3
        dy = 1.5
        shift = SQRT3 / 2
        for r in range(rows):
            for c in range(cols):
                x = c * dx
                if r % 2 == 1:
                    x += shift
                y = r * dy
                pos.append([x, y])
        pos = np.array(pos)
        # Center it
        pos = pos - pos.mean(axis=0)
        alphas = np.full(len(pos), np.pi / 6) # 30 degrees
        return pos, alphas

    # Structure 2: Honeycomb 4x3 (Pointy Top)
    # Pointy top hexagons: angle 0 deg.
    # Spacing: dx = 1.5, dy = sqrt(3). Shift odd rows by 0.75.
    def make_honeycomb_pointy(rows, cols):
        pos = []
        dx = 1.5
        dy = SQRT3
        shift = 0.75
        for r in range(rows):
            for c in range(cols):
                x = r * dx
                if c % 2 == 1: # Wait, standard honeycomb logic depends on orientation.
                    # Let's stick to the previous logic which worked:
                    pass
                # Re-implementing standard honeycomb for pointy top:
                # y = r * dy
                # x = c * dx + (dy * 0.5 if r % 2 else 0) -- No, spacing is different.
                # For pointy top:
                # Horizontal distance between centers in same row: sqrt(3)
                # Vertical distance between rows: 1.5
                # Horizontal shift for odd rows: sqrt(3)/2
                # Wait, the previous code had:
                # dx = 1.5, dy = SQRT3. This is for flat-top? No.
                # Let's use the robust generator from Inspiration 1.
                pass
        # Reusing Inspiration 1's logic which is known to work
        pos = []
        if cols >= rows: # Assume flat top if wider? No, let's just generate both explicitly.
            pass
        
        # Correct Pointy Top Honeycomb:
        # Angle 0.
        # Horizontal spacing: sqrt(3)
        # Vertical spacing: 1.5
        # Shift: sqrt(3)/2
        # This is actually the same spacing as Flat Top but rotated?
        # No.
        # Flat Top (angle 30):
        #   Width = 2*side, Height = sqrt(3)*side
        #   Horiz spacing = sqrt(3)*side
        #   Vert spacing = 1.5*side
        # Pointy Top (angle 0):
        #   Width = sqrt(3)*side, Height = 2*side
        #   Horiz spacing = 1.5*side
        #   Vert spacing = sqrt(3)*side
        
        # So for Pointy Top:
        dx = 1.5
        dy = SQRT3
        shift = 0.75 # sqrt(3)/2? No, 1.5/2 = 0.75.
        # Let's check.
        # If I place (0,0) and (1.5, sqrt(3)/2) -> dist = sqrt(2.25 + 0.75) = sqrt(3). Correct.
        # So shift is sqrt(3)/2? No, x-shift is 0.75.
        # Let's just use the function from inspiration 1.
        
        pos = []
        for r in range(rows):
            for c in range(cols):
                x = c * SQRT3
                if r % 2 == 1:
                    x += SQRT3 / 2
                y = r * 1.5
                pos.append([x, y])
        pos = np.array(pos)
        pos = pos - pos.mean(axis=0)
        alphas = np.full(len(pos), 0.0)
        return pos, alphas

    # Structure 3: Hex Cluster (1 center, 6 ring 1, 5 ring 2)
    def make_hex_cluster():
        pos = [[0.0, 0.0]]
        alphas = [0.0]
        # Ring 1: 6 hexagons at distance 2*sqrt(3)? No, touching distance is 2*side?
        # For flat top (30 deg), distance to touch is 2*sqrt(3)?
        # Let's check.
        # Two flat top hexagons side by side.
        # Center 1: (0,0). Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
        # Wait, angle 30 deg.
        # Vertex 0: (cos(30), sin(30)) = (sqrt(3)/2, 0.5).
        # Vertex 1: (cos(90), sin(90)) = (0, 1).
        # Distance between centers for touching:
        # The "radius" to a vertex is 1. The "radius" to an edge (apothem) is sqrt(3)/2.
        # If aligned same way, they touch at vertices? No, edges.
        # Distance = 2 * apothem = sqrt(3)?
        # Let's verify.
        # If I move (sqrt(3), 0) right.
        # Left hex rightmost vertex: (sqrt(3)/2, 0.5) and (sqrt(3)/2, -0.5)? No.
        # Vertices:
        # 0: (sqrt(3)/2, 0.5)
        # 1: (0, 1)
        # 2: (-sqrt(3)/2, 0.5)
        # 3: (-sqrt(3)/2, -0.5)
        # 4: (0, -1)
        # 5: (sqrt(3)/2, -0.5)
        # Rightmost x is sqrt(3)/2.
        # If I place next center at (sqrt(3), 0), its leftmost x is sqrt(3) - sqrt(3)/2 = sqrt(3)/2.
        # So they touch at x=sqrt(3)/2.
        # So distance is sqrt(3).
        
        d1 = SQRT3
        for k in range(6):
            a = k * np.pi / 3
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
            alphas.append(0.0)
        
        # Ring 2: 5 hexagons?
        # In a hexagonal grid, ring 2 has 12 hexagons.
        # We only need 5 more to reach 12.
        # Let's pick 5 from the 12 positions in ring 2.
        # Ring 2 positions are at distance 2*sqrt(3) along axes and sqrt(3)*sqrt(3)=3 along diagonals?
        # Let's just generate all 12 and pick 5.
        ring2_pos = []
        # Directions: 6 axial, 6 diagonal?
        # Axial: (2*sqrt(3), 0), (sqrt(3), sqrt(3)), (0, 2*sqrt(3)), (-sqrt(3), sqrt(3)), (-2*sqrt(3), 0), (-sqrt(3), -sqrt(3)), (0, -2*sqrt(3)), (sqrt(3), -sqrt(3))
        # Wait, that's 8.
        # Actually, ring 2 in a hex grid has 12 cells.
        # Let's just use the positions from the "flower" pattern.
        # 1 center, 6 neighbors, 12 next neighbors.
        # We take 5 of the 12 next neighbors.
        
        # Let's just use the 3x4 honeycomb as the primary candidate, it's dense.
        # And the 4x3.
        # And the cluster.
        
        # For the cluster, let's just place 5 more in a rough circle.
        d2 = 2 * SQRT3
        for k in range(5):
            a = k * 2 * np.pi / 5 + np.pi / 5
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
            alphas.append(0.0)
            
        pos = np.array(pos)
        pos = pos - pos.mean(axis=0)
        return pos, np.array(alphas)

    # Generate initial configurations
    configs = []
    
    # 1. Honeycomb 3x4 (Flat Top)
    pos_34, alphas_34 = make_honeycomb_flat(3, 4)
    configs.append((pos_34, alphas_34))
    
    # 2. Honeycomb 4x3 (Pointy Top)
    pos_43, alphas_43 = make_honeycomb_pointy(4, 3)
    configs.append((pos_43, alphas_43))
    
    # 3. Hex Cluster
    pos_cl, alphas_cl = make_hex_cluster()
    configs.append((pos_cl, alphas_cl))
    
    best_side = float('inf')
    best_result = None

    # Search over outer hexagon rotation
    # The optimal rotation is likely one of the symmetry angles (0, 30, 60...)
    # But we can do a fine search.
    
    for pos, alphas in configs:
        if time.time() > deadline - 5.0:
            break
            
        # Try a range of outer rotations
        # We can optimize the outer rotation for a fixed inner config.
        # The function side(theta) is piecewise linear (max of linear functions).
        # We can find the minimum by checking the breakpoints or just sampling.
        
        # Sample 360 angles
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        
        for theta in angles:
            side, valid = evaluate_config(pos, alphas, (0.0, 0.0), theta)
            if valid and side < best_side:
                best_side = side
                best_result = (pos.copy(), alphas.copy(), (0.0, 0.0), theta)
                
        # Refine around the best angle found for this config
        if best_result is not None and best_result[0] is pos:
            best_theta = best_result[3]
            # Fine search
            fine_angles = np.linspace(best_theta - np.pi/180, best_theta + np.pi/180, 100)
            for theta in fine_angles:
                side, valid = evaluate_config(pos, alphas, (0.0, 0.0), theta)
                if valid and side < best_side:
                    best_side = side
                    best_result = (pos.copy(), alphas.copy(), (0.0, 0.0), theta)

    # If we found a result, use it.
    if best_result is not None:
        pos, alphas, center, theta = best_result
        inner = []
        for i in range(N):
            inner.append([float(pos[i, 0]), float(pos[i, 1]), float(np.degrees(alphas[i]))])
        center_out = (float(center[0]), float(center[1]))
        side = float(best_side)
        angle_out = float(np.degrees(theta))
        
        return inner, center_out, side, angle_out
    else:
        # Fallback: Large box
        inner = []
        for i in range(N):
            inner.append([0.0, 0.0, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

# EVOLVE-BLOCK-END
