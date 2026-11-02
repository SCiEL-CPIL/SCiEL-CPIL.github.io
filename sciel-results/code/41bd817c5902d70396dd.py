import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap_depth(v1, v2):
        """
        Returns min overlap depth (positive if overlapping, 0 if disjoint or touching).
        """
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return max(0.0, min_overlap)

    def compute_min_outer_side(hexes, outer_center, outer_angle_rad):
        """
        Computes the minimum side length of a regular hexagon centered at outer_center
        with rotation outer_angle_rad that contains all hexes in the list.
        """
        outer_verts_template = hex_vertices(0, 0, 1.0, outer_angle_rad)
        normals = get_normals(outer_verts_template)
        
        max_proj = 0.0
        for h in hexes:
            # Translate hex to be relative to outer center
            h_rel = h - np.array(outer_center)
            proj = h_rel @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # For a regular hexagon of side s, the distance from center to side (apothem) is s * sqrt(3) / 2
        # max_proj is the max distance of any point from center along a normal.
        # So apothem_required = max_proj
        # s * sqrt(3) / 2 = max_proj  => s = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns s_required + penalty * overlap.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate the 12 hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # Outer hexagon is centered at (0,0) with angle 0
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # Calculate minimal side length to contain all
        s_req = compute_min_outer_side(hexes, outer_center, outer_angle)
        
        # Calculate overlap penalty
        penalty = 0.0
        # Check overlaps only for pairs that are close
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap_depth(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap
        
        # We want to minimize s_req, but we must avoid overlap.
        # If penalty > 0, we add a large cost.
        # To make it smooth for optimizers, we can use a quadratic penalty or just a large linear one.
        # Linear penalty with high weight is standard for hard constraints in this context.
        if penalty > 1e-6:
            return s_req + penalty * 100.0
        else:
            return s_req

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons
    # The hexagons must fit inside the outer hexagon.
    # Max possible s is roughly 4-5.
    # So centers should be within [-3, 3] roughly.
    bounds = [
        (-3.0, 3.0),  # x1
        (-3.0, 3.0),  # y1
        (0.0, np.pi/3), # a1
        (-3.0, 3.0),  # x2
        (-3.0, 3.0),  # y2
        (0.0, np.pi/3) # a2
    ]
    
    best_s = np.inf
    best_params = None
    
    # 1. Global Search with Differential Evolution
    try:
        res = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=True, # Use L-BFGS-B to polish the result
            mutation=(0.5, 1.5),
            recombination=0.8
        )
        
        # Verify the result is valid (low penalty)
        final_obj = objective(res.x)
        # Re-check penalty specifically
        x1, y1, a1, x2, y2, a2 = res.x
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            hexes.append(h1)
            hexes.append(h2)
        
        valid = True
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                overlap = sat_overlap_depth(hexes[i], hexes[j])
                if overlap > 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s_val = compute_min_outer_side(hexes, (0,0), 0.0)
            if s_val < best_s:
                best_s = s_val
                best_params = res.x
                
    except Exception as e:
        pass

    # 2. Local Refinement from Multiple Starts
    # Generate several heuristic starting points
    starts = []
    
    # Heuristic 1: Symmetric ring
    # 12 hexes. 6-fold symmetry -> 2 per sector.
    # Place them at r1, r2 in the sector.
    for r1 in [1.5, 2.0, 2.5]:
        for r2 in [2.5, 3.0, 3.5]:
            for a1 in [0, np.pi/6, np.pi/3]:
                for a2 in [0, np.pi/6, np.pi/3]:
                    # x1, y1 for r1, angle 0 (sector axis)
                    # x2, y2 for r2, angle 0 (sector axis)
                    # Wait, the symmetry rotates the whole coordinate system.
                    # So base1 is at (r1, 0) with rotation a1.
                    # base2 is at (r2, 0) with rotation a2.
                    p = np.array([r1, 0, a1, r2, 0, a2])
                    starts.append(p)
                    
    # Heuristic 2: Randomized
    np.random.seed(42)
    for _ in range(20):
        p = np.random.uniform(bounds[0][0], bounds[0][1], size=6)
        # Constrain angles to [0, pi/3]
        p[2] = np.random.uniform(0, np.pi/3)
        p[5] = np.random.uniform(0, np.pi/3)
        starts.append(p)
        
    for p0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            res_local = minimize(
                objective,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            # Check validity
            x1, y1, a1, x2, y2, a2 = res_local.x
            base1 = hex_vertices(x1, y1, 1.0, a1)
            base2 = hex_vertices(x2, y2, 1.0, a2)
            hexes = []
            for k in range(6):
                angle = np.radians(60 * k)
                c = np.cos(angle)
                s_ = np.sin(angle)
                rot = np.array([[c, -s_], [s_, c]])
                h1 = base1 @ rot.T
                h2 = base2 @ rot.T
                hexes.append(h1)
                hexes.append(h2)
                
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    min1 = hexes[i].min(axis=0)
                    max1 = hexes[i].max(axis=0)
                    min2 = hexes[j].min(axis=0)
                    max2 = hexes[j].max(axis=0)
                    if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                        continue
                    overlap = sat_overlap_depth(hexes[i], hexes[j])
                    if overlap > 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                s_val = compute_min_outer_side(hexes, (0,0), 0.0)
                if s_val < best_s:
                    best_s = s_val
                    best_params = res_local.x
                    
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_params is None:
        # Return a trivial valid solution (e.g. large hexagon)
        # 12 hexes in a line? No, must be inside a hexagon.
        # Just place them in a grid inside a large hexagon.
        best_params = np.array([0, 0, 0, 0, 0, 0]) # This will overlap, but we need to return something.
        # Let's construct a valid one manually:
        # Center 1, Ring 6, Ring 5?
        # This is hard to do with 6-fold symmetry if N=12 (2 per orbit).
        # Let's just return a very large s with non-overlapping hexes.
        # Place 12 hexes in a circle of radius 5.
        inner = []
        for i in range(12):
            angle = i * np.pi / 6
            x = 5.0 * np.cos(angle)
            y = 5.0 * np.sin(angle)
            inner.append([x, y, 0.0])
        outer_center = (0.0, 0.0)
        outer_side = 10.0
        outer_angle_deg = 0.0
        return inner, outer_center, outer_side, outer_angle_deg

    x1, y1, a1, x2, y2, a2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        # Rotate (x1, y1)
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        # Rotate (x2, y2)
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
        inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
        
    outer_center = (0.0, 0.0)
    outer_side = best_s
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
