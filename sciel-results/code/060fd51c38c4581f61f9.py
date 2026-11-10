# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon.
    Uses a 4x3 lattice structure with a slight expansion to ensure strict separation.
    Optimizes the rotation of the inner cluster and the outer container.
    """
    start_time = time.time()
    DEADLINE = 600.0

    # 1. Define the base lattice for 12 hexagons (4x3 grid)
    # Axial coordinates for a 4x3 block
    axial_coords = []
    for r in range(3):
        for q in range(4):
            axial_coords.append((q, r))
    axial_coords = np.array(axial_coords, dtype=float)
    
    # Convert to Cartesian (Pointy-top unit hexagons)
    # x = sqrt(3) * (q + r/2), y = 1.5 * r
    q = axial_coords[:, 0]
    r = axial_coords[:, 1]
    base_x = np.sqrt(3) * (q + r / 2.0)
    base_y = 1.5 * r
    base_centers = np.column_stack([base_x, base_y])
    
    # Center the configuration
    base_centers -= np.mean(base_centers, axis=0)
    
    # 2. Apply scaling factor to ensure separation > 1e-6
    # Distance between centers in base lattice is sqrt(3).
    # Required gap > 1e-6.
    # New distance = k * sqrt(3).
    # Gap = (k-1)*sqrt(3) > 1e-6 => k > 1 + 1e-6/sqrt(3)
    gap_target = 1e-5  # Safer margin
    k = 1.0 + gap_target / np.sqrt(3)
    
    def get_inner_centers(angle_rad):
        """Returns centers of the 12 hexagons rotated by angle_rad and scaled by k."""
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad), np.cos(angle_rad)]])
        rotated = base_centers @ R.T
        return rotated * k

    def get_all_vertices(centers, inner_angle_rad):
        """Get all 72 vertices of the 12 hexagons."""
        vertices = []
        for i in range(12):
            cx, cy = centers[i]
            for j in range(6):
                theta = inner_angle_rad + np.radians(60 * j)
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                vertices.append([vx, vy])
        return np.array(vertices)

    def compute_min_containing_hex_side(vertices, outer_angle_rad):
        """
        Compute the minimum side length s of a regular hexagon with given angle
        that contains all vertices.
        The center is optimized to minimize the maximum projection.
        """
        # Normals for the outer hexagon edges
        normals = []
        for k_idx in range(6):
            theta = outer_angle_rad + np.pi / 6 + k_idx * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        # To minimize s, we need to find center c that minimizes:
        # s = (2/sqrt(3)) * max_{v, n} dot(v-c, n)
        # This is equivalent to finding the center of the minimal bounding box
        # in the transformed space, or simply using the centroid as a good heuristic
        # and then refining.
        
        # Heuristic: Center at the mean of vertices
        c = np.mean(vertices, axis=0)
        
        # Calculate required s for this center
        rel = vertices - c
        dots = rel @ normals.T  # (72, 6)
        max_dots = np.max(dots, axis=1)
        s = (2.0 / np.sqrt(3.0)) * np.max(max_dots)
        
        # Refine center: The optimal center c minimizes the range of projections.
        # For a regular hexagon, the optimal center is the average of the "supporting" vertices?
        # A simpler refinement: try small shifts around the mean.
        # Or use scipy minimize on c.
        
        def obj_c(c_vec):
            rel = vertices - c_vec
            dots = rel @ normals.T
            max_dots = np.max(dots, axis=1)
            return np.max(max_dots)
        
        # Optimize center
        res = minimize(obj_c, c, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 100})
        if res.success:
            c_opt = res.x
            rel = vertices - c_opt
            dots = rel @ normals.T
            max_dots = np.max(dots, axis=1)
            s_opt = (2.0 / np.sqrt(3.0)) * np.max(max_dots)
            if s_opt < s:
                s = s_opt
                c = c_opt
                
        return s, c

    # 3. Optimization Loop
    # Search over inner rotation and outer rotation
    # Due to symmetry, we only need to search 0 to 30 degrees for inner rotation?
    # The lattice is 4x3, not fully symmetric, so search 0 to 180.
    # Outer hexagon symmetry is 60 degrees.
    
    best_s = float('inf')
    best_inner_angle = 0.0
    best_outer_angle = 0.0
    best_centers = None
    best_outer_center = None
    
    # Coarse Grid Search
    # Inner angle: 0 to pi (180 deg)
    # Outer angle: 0 to pi/3 (60 deg)
    
    n_inner_steps = 36
    n_outer_steps = 12
    
    inner_angles = np.linspace(0, np.pi, n_inner_steps)
    outer_angles = np.linspace(0, np.pi / 3, n_outer_steps)
    
    for ia in inner_angles:
        if time.time() - start_time > DEADLINE:
            break
        centers = get_inner_centers(ia)
        vertices = get_all_vertices(centers, ia)
        
        for oa in outer_angles:
            if time.time() - start_time > DEADLINE:
                break
            s, c_out = compute_min_containing_hex_side(vertices, oa)
            if s < best_s:
                best_s = s
                best_inner_angle = ia
                best_outer_angle = oa
                best_centers = centers
                best_outer_center = c_out

    # Fine Grid Search around best
    if time.time() - start_time < DEADLINE - 10:
        range_inner = 0.05
        range_outer = 0.05
        n_fine = 20
        
        fine_inner = np.linspace(best_inner_angle - range_inner, best_inner_angle + range_inner, n_fine)
        fine_outer = np.linspace(best_outer_angle - range_outer, best_outer_angle + range_outer, n_fine)
        
        for ia in fine_inner:
            if time.time() - start_time > DEADLINE:
                break
            centers = get_inner_centers(ia)
            vertices = get_all_vertices(centers, ia)
            
            for oa in fine_outer:
                if time.time() - start_time > DEADLINE:
                    break
                s, c_out = compute_min_containing_hex_side(vertices, oa)
                if s < best_s:
                    best_s = s
                    best_inner_angle = ia
                    best_outer_angle = oa
                    best_centers = centers
                    best_outer_center = c_out

    # Construct Final Result
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = best_centers[i, 0]
        inner[i, 1] = best_centers[i, 1]
        inner[i, 2] = np.degrees(best_inner_angle)
        
    outer = np.array([best_outer_center[0], best_outer_center[1], np.degrees(best_outer_angle)])
    
    return inner, outer, best_s

# EVOLVE-BLOCK-END
