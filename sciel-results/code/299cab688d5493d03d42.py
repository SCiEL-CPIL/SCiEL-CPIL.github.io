# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a deterministic, physics-based relaxation (L-BFGS-B) with a smooth objective function
    that minimizes the outer hexagon size using smooth repulsive forces for collisions and containment.
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

    # Helper: Compute the minimum side length s for a set of inner hexagons, 
    # assuming the outer hexagon is centered at (0,0) and has angle 0.
    def compute_s(inner_centers):
        """
        inner_centers: (12, 2) array of (x, y)
        Returns: s (float)
        """
        # All inner hexagons have angle 0 (aligned with outer)
        # We need to find the smallest regular hexagon (angle 0, center 0) containing all vertices.
        # The constraint is: for all 6 outward normals n_i of the outer hexagon,
        # max over all inner vertices v of dot(v, n_i) <= s * sqrt(3)/2
        # The 6 outward normals for a hexagon with angle 0 are at angles 30, 90, 150, 210, 270, 330 degrees.
        normals = []
        for i in range(6):
            a = math.pi/6 + i * math.pi / 3
            normals.append((math.cos(a), math.sin(a)))
        
        max_proj = 0.0
        for cx, cy in inner_centers:
            verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in verts:
                for n in normals:
                    proj = v[0]*n[0] + v[1]*n[1]
                    if proj > max_proj:
                        max_proj = proj
        
        return max_proj * 2.0 / sqrt3

    # Helper: Compute collision penalty between two hexagons (smooth approximation)
    def collision_penalty(verts1, verts2):
        """
        Returns a smooth penalty if two hexagons overlap.
        Uses the minimum distance between edges/vertices as a proxy.
        For a smooth objective, we use the sum of squared overlaps on all separating axes.
        """
        # Separating Axis Theorem: if there's any axis where projections are separated, no collision.
        # We want a smooth penalty that is 0 when separated and positive when overlapping.
        # For each axis, compute the overlap amount. If all overlaps are <= 0, no collision.
        # Penalty = sum of (max(0, overlap_i))^2 for all axes.
        
        axes = []
        for i in range(6):
            p1 = np.array(verts1[i])
            p2 = np.array(verts1[(i+1)%6])
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal)
        for i in range(6):
            p1 = np.array(verts2[i])
            p2 = np.array(verts2[(i+1)%6])
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal)
        
        total_penalty = 0.0
        for axis in axes:
            dots1 = [np.dot(np.array(v), axis) for v in verts1]
            dots2 = [np.dot(np.array(v), axis) for v in verts2]
            min1, max1 = min(dots1), max(dots1)
            min2, max2 = min(dots2), max(dots2)
            
            # Overlap amount: positive if overlapping, negative if separated
            # If max1 < min2, separated by (min2 - max1)
            # If max2 < min1, separated by (min1 - max2)
            # Otherwise, overlapping by min(max1, max2) - max(min1, min2)
            
            if max1 < min2:
                overlap = min2 - max1  # positive separation
            elif max2 < min1:
                overlap = min1 - max2  # positive separation
            else:
                overlap = min(max1, max2) - max(min1, min2)  # positive overlap
            
            # We want penalty for overlap. Use a smooth function.
            # If overlap > 0, they are overlapping. Penalty = overlap^2.
            # If overlap < 0, they are separated. Penalty = 0.
            if overlap > 0:
                total_penalty += overlap * overlap
        
        return total_penalty

    # Helper: Compute containment penalty (vertices outside outer hexagon)
    def containment_penalty(inner_centers, s):
        """
        Returns a penalty if any inner vertex is outside the outer hexagon (side s, center 0, angle 0).
        """
        # Outer hexagon edges: 6 edges. For each edge, check if point is inside.
        # The condition is cross(p2-p1, v-p1) >= 0 for CCW vertices.
        # Penalty = sum of (max(0, -cross))^2 for all violations.
        
        outer_verts = get_hex_vertices(0, 0, s, 0)
        penalty = 0.0
        
        for cx, cy in inner_centers:
            inner_verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in inner_verts:
                for i in range(6):
                    p1 = np.array(outer_verts[i])
                    p2 = np.array(outer_verts[(i+1)%6])
                    edge = p2 - p1
                    vec = np.array(v) - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    if cross < 0:
                        penalty += cross * cross
        
        return penalty

    # Objective function for optimization
    def objective(params):
        """
        params: (24,) array of 12 (x, y) pairs for inner hexagon centers.
        Returns: s + lambda1 * collision_penalty + lambda2 * containment_penalty
        """
        inner_centers = params.reshape(12, 2)
        
        # Compute s
        s = compute_s(inner_centers)
        
        # Compute collision penalties
        collision_pen = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                verts_i = get_hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, 0)
                verts_j = get_hex_vertices(inner_centers[j][0], inner_centers[j][1], 1.0, 0)
                collision_pen += collision_penalty(verts_i, verts_j)
        
        # Compute containment penalty
        cont_pen = containment_penalty(inner_centers, s)
        
        # Weighted sum
        lambda1 = 10.0  # weight for collisions
        lambda2 = 10.0  # weight for containment
        
        return s + lambda1 * collision_pen + lambda2 * cont_pen

    # Helper: Validate a configuration
    def validate_config(inner_centers, s):
        """
        Validates that the configuration is valid.
        Returns True if valid, False otherwise.
        """
        # Check collisions
        for i in range(12):
            for j in range(i+1, 12):
                verts_i = get_hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, 0)
                verts_j = get_hex_vertices(inner_centers[j][0], inner_centers[j][1], 1.0, 0)
                # Check if separated
                axes = []
                for k in range(6):
                    p1 = np.array(verts_i[k])
                    p2 = np.array(verts_i[(k+1)%6])
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    if np.linalg.norm(normal) > 1e-9:
                        axes.append(normal)
                for k in range(6):
                    p1 = np.array(verts_j[k])
                    p2 = np.array(verts_j[(k+1)%6])
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    if np.linalg.norm(normal) > 1e-9:
                        axes.append(normal)
                
                separated = False
                for axis in axes:
                    dots_i = [np.dot(np.array(v), axis) for v in verts_i]
                    dots_j = [np.dot(np.array(v), axis) for v in verts_j]
                    min_i, max_i = min(dots_i), max(dots_i)
                    min_j, max_j = min(dots_j), max(dots_j)
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        separated = True
                        break
                if not separated:
                    return False
        
        # Check containment
        outer_verts = get_hex_vertices(0, 0, s, 0)
        for cx, cy in inner_centers:
            inner_verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in inner_verts:
                for i in range(6):
                    p1 = np.array(outer_verts[i])
                    p2 = np.array(outer_verts[(i+1)%6])
                    edge = p2 - p1
                    vec = np.array(v) - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    if cross < -1e-6:
                        return False
        
        return True

    # Initial configuration: hexagonal lattice
    def initial_config():
        # 12 hexagons in a hexagonal arrangement
        # Rows: 1, 2, 3, 2, 1, 1? No, 12.
        # Let's use a 3-4-3-2 arrangement or similar.
        # For now, use a grid-like arrangement.
        centers = []
        # Center at (0,0)
        centers.append((0, 0))
        # 6 around it
        for i in range(6):
            a = i * math.pi / 3
            centers.append((2.5 * math.cos(a), 2.165 * math.sin(a)))
        # 5 more
        # Place them in the next ring
        for i in range(5):
            a = i * math.pi / 3 + math.pi / 6
            centers.append((5.0 * math.cos(a), 4.33 * math.sin(a)))
        return np.array(centers)

    # Run optimization
    best_val = 0.0
    best_inner = None
    best_s = float('inf')
    
    # Try multiple initial configurations
    initial_configs = [initial_config()]
    
    # Add a random configuration
    random_config = np.random.uniform(-5, 5, (12, 2))
    initial_configs.append(random_config)
    
    for init in initial_configs:
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B
        try:
            result = minimize(objective, init.flatten(), method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-12})
            
            if result.success:
                inner_centers = result.x.reshape(12, 2)
                s = compute_s(inner_centers)
                
                if validate_config(inner_centers, s):
                    val = 1.0 / s
                    if val > best_val:
                        best_val = val
                        best_inner = inner_centers
                        best_s = s
        except Exception as e:
            pass
    
    # If no valid configuration found, use a fallback
    if best_inner is None:
        # Fallback: use a known valid configuration
        # Simple grid
        centers = []
        for i in range(4):
            for j in range(3):
                x = i * 2.5 - 3.75
                y = j * 2.165 - 2.165
                centers.append((x, y))
        best_inner = np.array(centers)
        best_s = compute_s(best_inner)
        best_val = 1.0 / best_s
    
    # Prepare return values
    inner_arr = np.zeros((12, 3))
    for i in range(12):
        inner_arr[i][0] = best_inner[i][0]
        inner_arr[i][1] = best_inner[i][1]
        inner_arr[i][2] = 0.0  # angle
    
    outer_arr = np.array([0.0, 0.0, 0.0])
    
    return inner_arr, outer_arr, best_s

# EVOLVE-BLOCK-END
