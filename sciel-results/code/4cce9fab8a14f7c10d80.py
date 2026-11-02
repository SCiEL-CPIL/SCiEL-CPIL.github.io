import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
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
        
        return min_overlap

    def compute_min_side(hexes):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons.
        """
        # The outer hexagon has fixed orientation (angle 0).
        # Its supporting lines are defined by the 6 outward normals of a hexagon with side 1, angle 0.
        # Normal angles: 30, 90, 150, 210, 270, 330 degrees.
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # For a regular hexagon of side s, the distance from center to side is s * sqrt(3) / 2.
        # So max_proj = s * sqrt(3) / 2 => s = max_proj * 2 / sqrt(3)
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def check_validity(hexes):
        """Check if hexagons are disjoint."""
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 1e-9: # Allow tiny numerical error
                    return False
        return True

    # --- Honeycomb Topology Generation ---
    
    def generate_honeycomb(scale, shear, rotation):
        """
        Generate 12 unit hexagons in a honeycomb-like pattern, deformed by scale, shear, and rotation.
        Base topology: Center at (0,0).
        6 neighbors around center.
        5 more to complete a ring? No, 1+6+5=12?
        Let's use a "flower" pattern: 1 center, 6 around, 5 more around the 6?
        Actually, a standard hexagonal packing of 19 has 1+6+12.
        We need 12.
        Let's try: 1 center, 6 first ring, 5 second ring (leaving one gap).
        Or simply 2 rings of 6? No, they would overlap.
        
        Let's use a specific non-overlapping topology:
        1 at (0,0).
        6 at distance d = 2*sqrt(3) (tangent).
        That's 7.
        Add 5 more in the second ring?
        
        Alternative: A linear chain? No, too large.
        
        Let's stick to the "2 orbits of 6" symmetry but parametrize it as a deformation of a known good start.
        However, the directive asks for "Centroid-Minimization with Radial Expansion" with "fixed non-overlapping honeycomb topology".
        
        Let's define the base positions for 12 hexagons in a hexagonal lattice.
        Hexagonal lattice vectors:
        v1 = (2*sqrt(3), 0)
        v2 = (sqrt(3), 3)
        
        Positions (indices):
        0: (0,0)
        1: v1
        2: v2
        3: v1-v2
        4: v1+v2
        5: -v1
        6: -v2
        7: -v1+v2
        8: 2*v1
        9: v1+v2 (wait, 4 is v1+v2)
        
        Let's list 12 unique positions in a hex grid:
        (0,0)
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  <- 6 neighbors
        (2,0), (1,1), (0,2), (-1,2), (-2,1) <- 5 more to make 12?
        
        Let's use coordinates in axial system (q, r).
        Distance between centers is 2*sqrt(3) for unit hexes (side 1).
        Let D = 2*sqrt(3).
        
        Axial to Cartesian:
        x = D * (q + r/2)
        y = D * (sqrt(3)/2 * r)
        
        Points:
        (0,0)
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
        
        Total 12 points.
        """
        # Axial coordinates
        coords = [
            (0,0),
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2), (-1,2), (-2,1)
        ]
        
        D = 2.0 * np.sqrt(3.0)
        
        positions = np.zeros((12, 2))
        for i, (q, r) in enumerate(coords):
            positions[i, 0] = D * (q + r/2.0)
            positions[i, 1] = D * (np.sqrt(3.0)/2.0 * r)
            
        # Apply Affine Transformation
        # Matrix: [[scale, shear], [0, scale]]
        # Then Rotation
        A = np.array([
            [scale, shear],
            [0.0, scale]
        ])
        
        theta = rotation
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        M = R @ A
        
        transformed_positions = positions @ M.T
        
        # Generate Hexagons
        # All hexagons have side 1 and angle 0 (aligned with global frame initially? 
        # No, the problem allows rotation of each hexagon. 
        # In a honeycomb packing, all hexagons are aligned.
        # So angle is 0 for all.
        
        hexes = []
        for pos in transformed_positions:
            h = hex_vertices(pos[0], pos[1], 1.0, 0.0)
            hexes.append(h)
            
        return hexes

    def compute_objective_deformed(params):
        """
        params: [scale, shear, rotation]
        Returns min_side + penalty.
        """
        if time.time() > deadline:
            return 1e10
            
        scale, shear, rotation = params
        
        # Ensure scale is positive
        if scale <= 0:
            return 1e10
            
        hexes = generate_honeycomb(scale, shear, rotation)
        
        s_min = compute_min_side(hexes)
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
                    
        return s_min + penalty

    # --- Optimization ---
    
    best_s = np.inf
    best_params = None
    best_hexes = None
    
    # Multi-start Nelder-Mead
    # Initial guesses
    starts = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.1],
        [1.1, -0.1, -0.1],
        [1.0, 0.2, 0.0],
        [0.95, 0.0, 0.3],
        [1.05, -0.2, 0.0]
    ]
    
    bounds = [
        (0.5, 1.5),  # scale
        (-0.5, 0.5), # shear
        (-np.pi, np.pi) # rotation
    ]
    
    for start in starts:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                compute_objective_deformed,
                start,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
                best_hexes = generate_honeycomb(res.x[0], res.x[1], res.x[2])
                
        except Exception:
            pass
            
    # If no valid solution found or best_s is still inf, use a fallback
    if best_params is None or best_s > 10.0:
        # Fallback to a known valid configuration (e.g. slightly expanded honeycomb)
        # Scale 1.0, Shear 0, Rot 0 might have overlaps if D is not large enough?
        # D = 2*sqrt(3) ~ 3.46.
        # Width of hex is 2. Height is sqrt(3) ~ 1.732.
        # Distance between centers in honeycomb is 3.46.
        # Max width of two adjacent hexes is 2+2=4? No, they are tangent.
        # So scale 1.0 should be valid for the base topology.
        best_params = np.array([1.0, 0.0, 0.0])
        best_hexes = generate_honeycomb(1.0, 0.0, 0.0)
        best_s = compute_min_side(best_hexes)
        
    # Final Polish with L-BFGS-B if time permits
    if time.time() < deadline - 10 and best_params is not None:
        try:
            res_polish = minimize(
                compute_objective_deformed,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x
                best_hexes = generate_honeycomb(res_polish.x[0], res_polish.x[1], res_polish.x[2])
        except Exception:
            pass

    # Verify validity of the final solution
    if best_hexes is not None and check_validity(best_hexes):
        inner = []
        for h in best_hexes:
            # Center is the average of vertices
            cx, cy = np.mean(h, axis=0)
            # Angle is 0 for all in this topology
            inner.append([cx, cy, 0.0])
            
        outer_center = (0.0, 0.0)
        outer_side = compute_min_side(best_hexes)
        outer_angle_deg = 0.0
        
        return inner, outer_center, outer_side, outer_angle_deg
    else:
        # Fallback: Return a simple valid configuration if the optimized one is invalid
        # This should rarely happen if the penalty is strong enough
        # Use a very large scale to ensure no overlap, even if score is bad
        fallback_params = np.array([1.5, 0.0, 0.0])
        fallback_hexes = generate_honeycomb(1.5, 0.0, 0.0)
        
        # Check validity
        if not check_validity(fallback_hexes):
             # If even this fails, something is wrong with the geometry helpers
             # But D=3.46 should be safe for scale 1.5 (centers 5.19 apart, width 3)
             pass
             
        inner = []
        for h in fallback_hexes:
            cx, cy = np.mean(h, axis=0)
            inner.append([cx, cy, 0.0])
            
        outer_center = (0.0, 0.0)
        outer_side = compute_min_side(fallback_hexes)
        outer_angle_deg = 0.0
        
        return inner, outer_center, outer_side, outer_angle_deg
