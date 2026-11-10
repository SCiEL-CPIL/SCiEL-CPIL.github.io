# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a simulated annealing approach with vectorized collision checks.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    # Constants
    sqrt3 = math.sqrt(3)
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return verts

    # Precompute unit hexagon vertices for angle 0 (since we assume angle 0 for all inner hexes initially)
    # Actually, we might allow rotation later, but let's start with angle 0 for speed.
    # If we allow rotation, we need to recompute vertices.
    # Let's stick to angle 0 for now to maximize speed, as rotation might not be necessary for optimal packing of identical regular hexagons.
    # Wait, the prompt says "angle_deg" is part of the return.
    # If we fix angle to 0, we can precompute the relative vertices.
    
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    # Precompute edge normals for unit hexagon (angle 0)
    # Edges: v1-v0, v2-v1, ...
    edges = np.roll(unit_hex_verts, -1, axis=0) - unit_hex_verts
    # Normal is perpendicular to edge. For CCW polygon, outward normal is (dy, -dx) or (-dy, dx)?
    # Let's check.
    # v0=(1,0), v1=(0.5, 0.866). Edge = (-0.5, 0.866).
    # Normal should point outward.
    # The center is (0,0). Midpoint of edge is (0.75, 0.433).
    # Vector from center to midpoint is (0.75, 0.433).
    # Dot product of normal and this vector should be positive.
    # n = (0.866, 0.5). Dot = 0.75*0.866 + 0.433*0.5 = 0.649 + 0.216 > 0.
    # So normal = (edge_y, -edge_x) ?
    # edge_y = 0.866, -edge_x = 0.5. Yes.
    # So normals = (edges[:, 1], -edges[:, 0])
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    # Normalize them
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    
    # Helper: Vectorized check for separation of two sets of vertices
    # verts1: (N, 2), verts2: (M, 2)
    # axes: (K, 2)
    # Returns True if separated on any axis
    def are_separated(verts1, verts2, axes, eps=1e-6):
        # Project verts1 onto axes: (N, K)
        proj1 = verts1 @ axes.T
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        
        # Project verts2 onto axes: (M, K)
        proj2 = verts2 @ axes.T
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # Check if max1 < min2 - eps OR max2 < min1 - eps for any axis
        # Separated if (max1 < min2 - eps) OR (max2 < min1 - eps)
        sep1 = max1 < min2 - eps
        sep2 = max2 < min1 - eps
        
        return np.any(sep1) or np.any(sep2)

    def is_point_in_hexagon(point, hex_verts, eps=1e-6):
        px, py = point
        for i in range(len(hex_verts)):
            p1 = np.array(hex_verts[i])
            p2 = np.array(hex_verts[(i+1)%len(hex_verts)])
            edge = p2 - p1
            vec = np.array([px, py]) - p1
            cross = edge[0]*vec[1] - edge[1]*vec[0]
            if cross < -eps:
                return False
        return True

    def min_s_for_hexagons(inner_hexes, angle_deg=0):
        # inner_hexes: list of (cx, cy, ang)
        # We assume ang=0 for all, so we can use precomputed unit_hex_verts
        # Gather all vertices
        all_verts = []
        for cx, cy, ang in inner_hexes:
            # If ang is not 0, we'd need to rotate. But we fix ang=0.
            verts = unit_hex_verts + np.array([cx, cy])
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        
        # Normals for outer hexagon (angle 0)
        # Outward normals at angles 30, 90, 150, 210, 270, 330
        outer_normals = []
        base_angle = math.radians(angle_deg)
        for i in range(6):
            normal_angle = base_angle + math.pi/6 + i * math.pi / 3
            outer_normals.append([math.cos(normal_angle), math.sin(normal_angle)])
        outer_normals = np.array(outer_normals)
        
        # Project all vertices onto outer normals
        projs = all_verts @ outer_normals.T
        max_proj = np.max(projs, axis=0)
        
        # s * sqrt(3)/2 >= max_proj
        # s >= max_proj * 2 / sqrt(3)
        return np.max(max_proj) * 2.0 / sqrt3

    def score_config(inner_hexes):
        # inner_hexes: list of [cx, cy, ang]
        # We assume ang=0.
        
        # 1. Check collisions
        # Get vertices for each hex
        verts_list = []
        for cx, cy, ang in inner_hexes:
            verts = unit_hex_verts + np.array([cx, cy])
            verts_list.append(verts)
        
        # Check all pairs
        for i in range(12):
            for j in range(i+1, 12):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                # Axes are normals of v1 and v2. Since they are same shape and orientation, normals are same.
                # We just need to check against the 6 unit normals.
                if not are_separated(v1, v2, normals, eps=1e-6):
                    return -1, None, None
        
        # 2. Calculate s
        s = min_s_for_hexagons(inner_hexes, angle_deg=0)
        
        # 3. Check containment
        # Outer hex centered at (0,0) with side s and angle 0.
        # Vertices of outer hex:
        outer_verts = []
        for i in range(6):
            a = i * math.pi / 3
            outer_verts.append([s * math.cos(a), s * math.sin(a)])
        outer_verts = np.array(outer_verts)
        
        # Check if all inner vertices are inside outer hex
        for verts in verts_list:
            for v in verts:
                if not is_point_in_hexagon(v, outer_verts, eps=1e-6):
                    return -1, None, None
        
        return 1.0/s, inner_hexes, s

    # Initial configuration
    # Hexagonal lattice
    dx = 2.5
    dy = sqrt3
    
    def generate_initial():
        init = []
        # 3 rows of 4? Or 4 rows of 3?
        # Let's try a 3x4 grid.
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2
                init.append([x, y, 0.0])
        return init

    best_val = 0.0
    best_inner = None
    best_s = float('inf')
    
    def simulated_annealing(init_inner, max_iter=10000):
        current_inner = [list(h) for h in init_inner]
        current_val, _, _ = score_config(current_inner)
        
        if current_val <= 0:
            # Try to fix initial config
            for _ in range(50):
                for i in range(12):
                    current_inner[i][0] += random.uniform(-1, 1)
                    current_inner[i][1] += random.uniform(-1, 1)
                current_val, _, _ = score_config(current_inner)
                if current_val > 0:
                    break
            if current_val <= 0:
                return -1, None, None

        best_local_val = current_val
        best_local_inner = [h[:] for h in current_inner]
        
        T = 1.0
        cooling = 0.999
        step_size = 0.5
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 11)
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            
            new_inner = [h[:] for h in current_inner]
            new_inner[idx][0] += dx
            new_inner[idx][1] += dy
            
            new_val, _, _ = score_config(new_inner)
            
            if new_val > current_val:
                current_inner = new_inner
                current_val = new_val
                if current_val > best_local_val:
                    best_local_val = current_val
                    best_local_inner = [h[:] for h in current_inner]
            else:
                # Accept worse state with probability exp(-delta / T)
                if new_val > 0:
                    delta = current_val - new_val
                    if random.random() < math.exp(-delta / T):
                        current_inner = new_inner
                        current_val = new_val
                
            T *= cooling
            if T < 0.001:
                break
        
        return best_local_val, best_local_inner, min_s_for_hexagons(best_local_inner)

    # Run simulated annealing
    init = generate_initial()
    val, inner, s = simulated_annealing(init)
    
    if val > best_val:
        best_val = val
        best_inner = inner
        best_s = s

    # If still poor, try random restarts
    if best_val < 1.5:
        for _ in range(10):
            if time.time() > deadline:
                break
            rand_inner = []
            for _ in range(12):
                rand_inner.append([random.uniform(-5, 5), random.uniform(-5, 5), 0.0])
            val, inner, s = simulated_annealing(rand_inner, max_iter=2000)
            if val > best_val:
                best_val = val
                best_inner = inner
                best_s = s

    # Finalize
    if best_inner is None:
        # Fallback
        best_inner = generate_initial()
        val, _, best_s = score_config(best_inner)
        best_val = val if val > 0 else 0.1

    # Shift inner hexes so centroid is (0,0) for consistency with s calculation?
    # The s calculation in score_config assumes center (0,0).
    # So we must return inner hexes that are centered at (0,0) relative to the outer hex.
    centers = np.array([(cx, cy) for cx, cy, _ in best_inner])
    centroid = np.mean(centers, axis=0)
    final_inner = [(cx - centroid[0], cy - centroid[1], ang) for cx, cy, ang in best_inner]
    
    # Recalculate s for final_inner to be safe
    final_val, _, final_s = score_config(final_inner)
    if final_val <= 0:
        final_s = 10.0
        final_val = 0.1

    inner_ret = np.array(final_inner)
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, final_s

# EVOLVE-BLOCK-END
