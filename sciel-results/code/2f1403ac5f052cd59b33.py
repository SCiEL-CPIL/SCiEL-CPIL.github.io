# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
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
    EPS_SEP = 1e-6
    EPS_IN = 1e-6

    # Precompute axes for unit hexagon at angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Edge 0: v0(1,0) to v1(0.5, sqrt(3)/2)
    # Edge vector: (-0.5, sqrt(3)/2)
    # Normal (left): (-sqrt(3)/2, -0.5) -> Outward?
    # Let's check. Center (0,0). Midpoint of edge 0: (0.75, sqrt(3)/4).
    # Dot product with normal (-sqrt(3)/2, -0.5):
    # 0.75*(-sqrt(3)/2) + sqrt(3)/4*(-0.5) = -sqrt(3)/2 * 0.75 - sqrt(3)/8 = negative.
    # So inward normal is (-sqrt(3)/2, -0.5). Outward is (sqrt(3)/2, 0.5).
    # Actually, let's just compute normals from vertices.
    
    def get_unit_hex_axes(angle_deg=0):
        """Returns 6 outward normals for a unit hexagon rotated by angle_deg."""
        rad = math.radians(angle_deg)
        axes = []
        for i in range(6):
            # Vertex i
            a_i = rad + i * math.pi / 3
            v_i = np.array([math.cos(a_i), math.sin(a_i)])
            # Vertex i+1
            a_ip1 = rad + (i+1) * math.pi / 3
            v_ip1 = np.array([math.cos(a_ip1), math.sin(a_ip1)])
            
            edge = v_ip1 - v_i
            # Normal perpendicular to edge.
            # We want the outward normal.
            # For a CCW polygon, the outward normal is to the right of the edge vector (p2-p1).
            # Right normal of (dx, dy) is (dy, -dx).
            n = np.array([edge[1], -edge[0]])
            # Normalize
            n = n / np.linalg.norm(n)
            # Ensure it points away from origin (0,0)
            # Midpoint of edge is (v_i + v_ip1) / 2
            mid = (v_i + v_ip1) / 2.0
            if np.dot(n, mid) < 0:
                n = -n
            axes.append(n)
        return np.array(axes)

    # Precompute axes for angle 0 (most common case)
    AXES_0 = get_unit_hex_axes(0.0)
    
    # Rotation matrix for angle theta
    def rot_matrix(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]])

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return verts

    def are_hexagons_separated_fast(c1, a1, c2, a2):
        """
        c1, c2: centers (2,)
        a1, a2: angles in radians
        """
        # If angles are close to 0, use precomputed axes?
        # For simplicity and speed, we can just compute the 12 axes on the fly or cache.
        # Given 12 hexes, 66 pairs. 12 axes per pair.
        # Let's just compute.
        
        # Axes for hex 1
        axes1 = get_unit_hex_axes(math.degrees(a1))
        # Axes for hex 2
        axes2 = get_unit_hex_axes(math.degrees(a2))
        
        # Combine axes
        axes = np.vstack([axes1, axes2])
        
        # Project vertices of hex 1
        # Vertices of hex 1: c1 + R1 * V
        R1 = rot_matrix(a1)
        V_unit = np.array([get_hex_vertices(0, 0, 1, 0)]) # (6,2)
        verts1 = c1 + (V_unit @ R1.T)[0] # (6,2)
        
        # Vertices of hex 2
        R2 = rot_matrix(a2)
        verts2 = c2 + (V_unit @ R2.T)[0] # (6,2)
        
        # Project onto axes
        # proj1 = verts1 @ axes.T  (6, 12)
        proj1 = verts1 @ axes.T
        proj2 = verts2 @ axes.T
        
        min1 = proj1.min(axis=0)
        max1 = proj1.max(axis=0)
        min2 = proj2.min(axis=0)
        max2 = proj2.max(axis=0)
        
        # Check separation
        # Separated if max1 < min2 - EPS or max2 < min1 - EPS
        sep1 = max1 < min2 - EPS_SEP
        sep2 = max2 < min1 - EPS_SEP
        
        if np.any(sep1) or np.any(sep2):
            return True
            
        return False

    def min_s_for_hexagons_fast(centers, angles):
        """
        centers: (12, 2)
        angles: (12,) in radians
        Returns min s for outer hex centered at (0,0) angle 0.
        """
        all_verts = []
        for i in range(len(centers)):
            R = rot_matrix(angles[i])
            V_unit = np.array([get_hex_vertices(0, 0, 1, 0)])[0] # (6,2)
            verts = centers[i] + (V_unit @ R.T)
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts) # (72, 2)
        
        # Normals for outer hex (angle 0)
        # Outward normals at 30, 90, 150, 210, 270, 330
        normals = []
        for i in range(6):
            a = math.radians(30 + i * 60)
            normals.append(np.array([math.cos(a), math.sin(a)]))
        normals = np.array(normals) # (6, 2)
        
        # Projections
        projs = all_verts @ normals.T # (72, 6)
        max_proj = projs.max(axis=0) # (6,)
        
        # s * sqrt(3)/2 >= max_proj
        # s >= max_proj * 2 / sqrt(3)
        s = np.max(max_proj) * 2.0 / sqrt3
        return s

    def validate_config_fast(centers, angles, s, outer_center=(0,0), outer_angle=0):
        """
        centers: (12, 2)
        angles: (12,) radians
        s: float
        """
        # 1. Check collisions
        for i in range(12):
            for j in range(i+1, 12):
                if not are_hexagons_separated_fast(centers[i], angles[i], centers[j], angles[j]):
                    return False
        
        # 2. Check containment
        # Outer hex vertices
        outer_verts = get_hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
        
        # Inner vertices
        all_inner_verts = []
        for i in range(12):
            R = rot_matrix(angles[i])
            V_unit = np.array([get_hex_vertices(0, 0, 1, 0)])[0]
            verts = centers[i] + (V_unit @ R.T)
            all_inner_verts.append(verts)
        all_inner_verts = np.vstack(all_inner_verts)
        
        # Check each vertex against outer hex edges
        # Outer hex is CCW. Interior is left of edge.
        # cross(p2-p1, v-p1) >= -EPS
        for i in range(6):
            p1 = np.array(outer_verts[i])
            p2 = np.array(outer_verts[(i+1)%6])
            edge = p2 - p1
            # v - p1 for all vertices
            vecs = all_inner_verts - p1
            # cross(edge, vecs)
            # edge[0]*vecs[:,1] - edge[1]*vecs[:,0]
            crosses = edge[0]*vecs[:,1] - edge[1]*vecs[:,0]
            if np.any(crosses < -EPS_IN):
                return False
        
        return True

    def score_config_fast(centers, angles):
        """
        centers: (12, 2)
        angles: (12,) radians
        Returns (val, shifted_centers, s) or (-1, None, None)
        """
        # Center the configuration
        centroid = np.mean(centers, axis=0)
        shifted_centers = centers - centroid
        shifted_angles = angles
        
        s = min_s_for_hexagons_fast(shifted_centers, shifted_angles)
        
        if validate_config_fast(shifted_centers, shifted_angles, s, (0,0), 0):
            return 1.0/s, shifted_centers, s
        else:
            return -1, None, None

    # Initial configuration
    # Hexagonal lattice
    dx = 2.5
    dy = sqrt3
    
    def generate_initial():
        # 12 hexes in a 3-4-3-2 pattern? Or 4-4-4?
        # Let's try a 3x4 grid
        init = []
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2.0
                init.append([x, y, 0.0])
        return np.array(init)[:, :2], np.array(init)[:, 2]

    best_val = 0.0
    best_centers = None
    best_angles = None
    best_s = float('inf')

    def simulated_annealing(init_centers, init_angles, max_iter=5000):
        current_centers = init_centers.copy()
        current_angles = init_angles.copy()
        current_val, _, _ = score_config_fast(current_centers, current_angles)
        
        if current_val <= 0:
            # Try to fix
            for _ in range(50):
                idx = random.randint(0, 11)
                current_centers[idx] += np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                current_val, _, _ = score_config_fast(current_centers, current_angles)
                if current_val > 0:
                    break
            if current_val <= 0:
                return -1, None, None, None

        best_local_val = current_val
        best_local_centers = current_centers.copy()
        best_local_angles = current_angles.copy()
        
        T = 0.5
        cooling = 0.999
        step_size = 0.5
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Random move
            idx = random.randint(0, 11)
            move_type = random.random()
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            
            if move_type < 0.8:
                # Move position
                new_centers[idx] += np.array([random.uniform(-step_size, step_size), random.uniform(-step_size, step_size)])
            else:
                # Rotate
                new_angles[idx] += random.uniform(-0.1, 0.1)
                
            new_val, _, _ = score_config_fast(new_centers, new_angles)
            
            if new_val > current_val:
                current_centers = new_centers
                current_angles = new_angles
                current_val = new_val
                if current_val > best_local_val:
                    best_local_val = current_val
                    best_local_centers = current_centers.copy()
                    best_local_angles = current_angles.copy()
            else:
                # Accept worse with prob
                if new_val > 0:
                    delta = current_val - new_val
                    if random.random() < math.exp(-delta / T):
                        current_centers = new_centers
                        current_angles = new_angles
                        current_val = new_val
                # If new_val is -1 (invalid), reject
                
            T *= cooling
            if T < 1e-4:
                T = 1e-4
                
            # Adaptive step size
            if it % 100 == 0 and it > 0:
                step_size *= 0.95
                
        return best_local_val, best_local_centers, best_local_angles, None

    # Run SA
    init_centers, init_angles = generate_initial()
    val, centers, angles, _ = simulated_annealing(init_centers, init_angles)
    
    if val > best_val:
        best_val = val
        best_centers = centers
        best_angles = angles
        best_s = min_s_for_hexagons_fast(centers, angles)

    # If SA didn't find anything good, try a few random restarts
    if best_val < 1.0:
        for _ in range(5):
            if time.time() > deadline:
                break
            rand_centers = np.random.uniform(-5, 5, (12, 2))
            rand_angles = np.zeros(12)
            val, centers, angles, _ = simulated_annealing(rand_centers, rand_angles, max_iter=1000)
            if val > best_val:
                best_val = val
                best_centers = centers
                best_angles = angles
                best_s = min_s_for_hexagons_fast(centers, angles)

    # Finalize
    if best_centers is None:
        # Fallback
        best_centers, best_angles = generate_initial()
        best_val, _, best_s = score_config_fast(best_centers, best_angles)
        if best_val < 0:
             # Last resort
             best_s = 10.0
             
    # Prepare return
    inner_ret = np.column_stack((best_centers, best_angles * 180.0 / math.pi))
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s

# EVOLVE-BLOCK-END
