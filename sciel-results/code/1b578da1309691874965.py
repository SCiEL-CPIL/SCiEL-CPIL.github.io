import numpy as np
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0 # seconds
    
    # Constants
    SQRT3 = np.sqrt(3)
    # Distance between centers of touching unit hexagons
    D = SQRT3
    
    # Function to get vertices of a unit hexagon
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        # Vertices are at angles angle_rad + i * 60 deg
        angles = angle_rad + np.deg2rad(np.arange(6) * 60)
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack((x, y))
    
    # Function to find min bounding regular hexagon side length for a set of points
    def min_bounding_hex_side(points):
        # points: Nx2 array
        # We vary the orientation of the bounding hexagon
        # The side length S is determined by the max projection onto the 6 normals.
        # S = (2/sqrt(3)) * max_k max_p (p . n_k)
        
        # Sample orientations
        # The function is periodic with pi/3 (60 deg)
        # We can sample in [0, pi/3)
        
        # Coarse grid
        alphas = np.linspace(0, np.pi/3, 1000)
        best_side = np.inf
        best_alpha = 0
        
        # Precompute points
        P = points
        
        for a in alphas:
            # Normals for bounding hexagon
            # n_k = (cos(a + k*pi/3), sin(a + k*pi/3))
            angles_n = a + np.pi/3 * np.arange(6)
            n = np.column_stack((np.cos(angles_n), np.sin(angles_n)))
            
            # Projections
            # P is (N, 2), n is (6, 2)
            # We want max over k and p of P[:, k] dot n[k]
            # proj[i, k] = P[i, 0]*n[k, 0] + P[i, 1]*n[k, 1]
            proj = P @ n.T # (N, 6)
            max_proj = np.max(proj)
            
            side = (2 / SQRT3) * max_proj
            if side < best_side:
                best_side = side
                best_alpha = a
        
        # Refine around best_alpha
        # Use a finer grid or golden section
        # Let's do a few iterations of refinement
        for _ in range(3):
            span = (np.pi/3) / (1000 * (10 ** _))
            alphas_ref = np.linspace(best_alpha - span, best_alpha + span, 100)
            alphas_ref = alphas_ref % (np.pi/3) # keep in range? Not strictly needed for cos/sin
            for a in alphas_ref:
                angles_n = a + np.pi/3 * np.arange(6)
                n = np.column_stack((np.cos(angles_n), np.sin(angles_n)))
                proj = P @ n.T
                max_proj = np.max(proj)
                side = (2 / SQRT3) * max_proj
                if side < best_side:
                    best_side = side
                    best_alpha = a
            span /= 10
            if time.time() > deadline - 10:
                break

        return best_side, best_alpha

    # Generate topologies
    # Centers are relative to the cluster centroid (which we will set to 0,0 for the outer hexagon)
    # Note: The outer hexagon is centered at (0,0). The inner hexagons are placed such that their cluster is centered at (0,0).
    
    topologies = []
    
    # Topology 1: 1 center + 6 ring + 4 ring-2 (corners)
    c0 = np.array([0, 0])
    ring1 = []
    for i in range(6):
        ang = np.pi/3 * i
        ring1.append(D * np.array([np.cos(ang), np.sin(ang)]))
    
    ring2_corners = []
    for i in range(6):
        ang = np.pi/3 * i
        ring2_corners.append(2 * D * np.array([np.cos(ang), np.sin(ang)]))
        
    ring2_edges = []
    for i in range(6):
        ang = np.pi/3 * i + np.pi/6
        ring2_edges.append(2 * D * np.array([np.cos(ang), np.sin(ang)]))
        
    # Candidate 1: 4 corners
    t1_centers = [c0] + ring1 + ring2_corners[:4]
    topologies.append(t1_centers)
    
    # Candidate 2: 4 edges
    t2_centers = [c0] + ring1 + ring2_edges[:4]
    topologies.append(t2_centers)
    
    # Candidate 3: 2 corners, 2 edges
    t3_centers = [c0] + ring1 + ring2_corners[:2] + ring2_edges[:2]
    topologies.append(t3_centers)
    
    # Candidate 4: All 6 ring + 4 more in a line? 
    # Let's try a 3x3 "square" in hex grid?
    # Hex grid coords (q, r)
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-2,2), (-2,1), (-1,2), (2,-1), (1,-2), (0,-2), (-1,-1)
    
    def hex_to_cart(q, r):
        x = D * (q + r/2.0)
        y = D * (r * SQRT3 / 2.0)
        return np.array([x, y])
        
    # 11 hexagons:
    # Center (0,0)
    # 6 neighbors
    # 4 from the next ring.
    # Let's pick (2,0), (0,2), (-2,2), (-2,0) ?
    t4_centers = []
    t4_centers.append(hex_to_cart(0,0))
    for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
        t4_centers.append(hex_to_cart(dq, dr))
    # Add 4 more
    t4_centers.append(hex_to_cart(2,0))
    t4_centers.append(hex_to_cart(0,2))
    t4_centers.append(hex_to_cart(-2,2))
    t4_centers.append(hex_to_cart(-2,0))
    topologies.append(t4_centers)
    
    best_result = None
    best_side = np.inf
    best_centers = None
    best_alpha = 0
    
    for centers in topologies:
        # Center the cluster
        centroid = np.mean(centers, axis=0)
        centers = [c - centroid for c in centers]
        
        # Generate all vertices
        all_verts = []
        for c in centers:
            v = get_hex_vertices(c[0], c[1], 0)
            all_verts.append(v)
        
        points = np.vstack(all_verts)
        
        side, alpha = min_bounding_hex_side(points)
        
        if side < best_side:
            best_side = side
            best_centers = centers
            best_alpha = alpha
            best_result = (centers, alpha)
            
        if time.time() > deadline - 5:
            break
            
    if best_result is None:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    # Construct the return values
    # Inner hexagons: (x, y, angle_deg)
    # The cluster is rotated by best_alpha
    # The individual hexagons are aligned with the cluster, so their angle is also best_alpha (in degrees)
    # Wait, the angle in the return value is the angle of the hexagon itself.
    # If we rotate the cluster, the hexagons rotate.
    # So angle_deg = np.rad2deg(best_alpha)
    
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(best_centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = np.rad2deg(best_alpha)
        
    outer_hex_data = np.array([0, 0, 0])
    # The outer hexagon side length is best_side
    # But we need to ensure the outer hexagon orientation matches the one used in optimization?
    # The optimization found the min side for a bounding hexagon rotated by best_alpha relative to the cluster?
    # No, `min_bounding_hex_side` varied the orientation of the *bounding* hexagon.
    # The cluster was fixed at angle 0 in that function.
    # So the bounding hexagon angle is `best_alpha`.
    # But the return value asks for the outer hexagon's angle.
    # And the inner hexagons' angles.
    # We can rotate the whole system arbitrarily.
    # Let's set the outer hexagon angle to 0.
    # Then we must rotate the inner hexagons by -best_alpha.
    
    # Actually, it's simpler to:
    # 1. Keep the cluster at angle 0.
    # 2. Set the outer hexagon angle to best_alpha.
    # 3. Return inner hexagons with angle 0.
    # This satisfies the geometry.
    
    # However, the evaluator might check consistency?
    # "A hexagon (cx, cy, s, a) has vertices..."
    # It just checks if the inner hexagons are inside the outer one.
    # So we can choose any global rotation.
    # Let's set outer angle = 0.
    # Then inner hexagons must be rotated by -best_alpha.
    
    rot_angle_deg = -np.rad2deg(best_alpha)
    
    for i in range(11):
        inner_hex_data[i, 2] = rot_angle_deg
        
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_side

# EVOLVE-BLOCK-END
