import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate the honeycomb lattice positions for 12 unit hexagons.
    # A regular honeycomb lattice of unit hexagons (side length 1) has:
    # - Horizontal distance between centers in same row: 2
    # - Vertical distance between rows: sqrt(3)
    # - Horizontal offset between adjacent rows: 1
    # 
    # Let's generate a 4x3 grid of centers and pick the 12 that form a compact cluster.
    # Actually, a "hexagonal" cluster of 12 is not a perfect hexagon (which would be 1, 7, 19...).
    # 12 is 7 + 5? Or a 3x4 rectangle in the lattice?
    # Let's create a 3x4 block of hexagons.
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    # Total 12.
    
    sqrt3 = math.sqrt(3)
    
    # Lattice vectors for "pointy-top" hexagons (vertex at (1,0))
    # Wait, the problem allows rotation. The relative orientation of inner hexagons 
    # to each other in a honeycomb packing is fixed (all parallel).
    # We can assume all inner hexagons have angle 0 relative to the lattice.
    # The global rotation of the whole cluster is handled by rotating the outer hexagon 
    # or the cluster. Since the objective is symmetric with respect to the outer hexagon's 
    # rotation relative to the cluster, we can fix the cluster orientation and optimize 
    # the outer hexagon's rotation.
    
    # Let's define the centers for a 3x4 grid of hexagons.
    # Using "flat-top" hexagons for the lattice calculation might be easier?
    # No, let's stick to the definition: Unit hexagon side 1.
    # If we pack them in a honeycomb, the centers form a triangular lattice.
    # Distance between centers in a honeycomb packing of unit hexagons is 2 * apothem? 
    # No, distance is 2 * inradius? 
    # Inradius r = sqrt(3)/2.
    # Distance between centers of adjacent hexagons in honeycomb = 2 * r = sqrt(3).
    # Wait.
    # Two unit hexagons touching side-to-side.
    # Center to center distance is 2 * (distance from center to edge) = 2 * (sqrt(3)/2) = sqrt(3).
    # So the lattice spacing is sqrt(3).
    
    # Let's generate centers on a triangular lattice with spacing sqrt(3).
    # Basis vectors:
    # v1 = (sqrt(3), 0)  -- Horizontal neighbor
    # v2 = (sqrt(3)/2, 3/2) -- Diagonal neighbor?
    # Let's check: |v2|^2 = 3/4 + 9/4 = 12/4 = 3. |v2| = sqrt(3). Correct.
    # Angle between v1 and v2: cos(theta) = (v1.v2)/(|v1||v2|) = (3/2) / 3 = 1/2. theta = 60 deg.
    # This forms a triangular lattice.
    
    # We want a compact cluster of 12 points.
    # A 3x4 rectangular block in this lattice?
    # Let's index by (i, j) where i is row, j is col.
    # Center(i, j) = i * v1 + j * v2 ?
    # Let's try to form a shape that is roughly circular to minimize bounding box.
    # 12 points.
    # Maybe a 3x4 grid is not the most compact.
    # A hexagonal number sequence: 1, 7, 19.
    # 12 is between 7 and 19.
    # Let's take the central 7 points of a larger hexagon and add 5 more?
    # Or just a 3x4 grid.
    # Let's generate a 4x3 grid (4 rows, 3 cols) or 3x4.
    
    centers = []
    # Let's use a "brick wall" pattern.
    # Row 0: x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3)  (4 hexes)
    # Row 1: x = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2, 7*sqrt(3)/2 (4 hexes)
    # Row 2: x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3) (4 hexes)
    
    # This forms a parallelogram.
    # Let's center it at (0,0).
    
    rows = 3
    cols = 4
    
    raw_centers = []
    for i in range(rows):
        for j in range(cols):
            # Shift row i by i * v2
            # v2 = (sqrt(3)/2, 3/2)
            x = j * sqrt(3) + i * (sqrt(3)/2)
            y = i * (3/2)
            raw_centers.append((x, y))
            
    # Center the cluster
    cx = sum(c[0] for c in raw_centers) / len(raw_centers)
    cy = sum(c[1] for c in raw_centers) / len(raw_centers)
    
    inner_centers = [(c[0] - cx, c[1] - cy) for c in raw_centers]
    
    # All inner hexagons have angle 0 (relative to each other).
    inner_angles = [0.0] * N
    
    # Function to calculate the minimum outer side length R 
    # given the outer hexagon center (ox, oy) and angle (theta)
    def calc_min_side(ox, oy, theta_deg):
        # Outer hexagon is centered at (ox, oy) with rotation theta.
        # We need the smallest R such that all inner hexagons (angle 0, centers inner_centers)
        # are inside the outer hexagon.
        
        # The condition "Point P is inside Hexagon(H)" is equivalent to:
        # max_k ( n_k . (P - C) ) <= R * r_in
        # where n_k are unit normal vectors to the sides of H,
        # C is the center of H,
        # r_in is the inradius of H (R * sqrt(3) / 2).
        
        # So, R * (sqrt(3)/2) >= max_k ( n_k . (P - C) )
        # R >= (2/sqrt(3)) * max_k ( n_k . (P - C) )
        
        # For the outer hexagon to contain an inner hexagon, it must contain all its vertices.
        # So we need R >= max_{inner_hex} max_{vertex v} ( (2/sqrt(3)) * max_k ( n_k . (v - C) ) )
        
        # Precompute unit vertices for inner hexagons (angle 0)
        # Vertices of unit hexagon at angle 0:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        
        verts_unit = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        # Precompute normals for outer hexagon
        theta_rad = math.radians(theta_deg)
        # Normals are perpendicular to edges.
        # For a hexagon with vertex at angle 0, edges are at angles 30, 90, ...
        # So normals are at 30, 90, 150, 210, 270, 330.
        # If we rotate the hexagon by theta, normals rotate by theta.
        # Normal angles: theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_required_r = 0.0
        
        for i in range(N):
            cx_i, cy_i = inner_centers[i]
            # Get all vertices of this inner hexagon
            # verts = center + verts_unit
            verts = np.array([cx_i, cy_i]) + verts_unit
            # Vector from outer center to each vertex
            # v_rel = verts - [ox, oy]
            v_rel = verts - np.array([ox, oy])
            
            # Calculate projection onto all normals
            # projections = v_rel @ normals.T
            projections = v_rel @ normals.T
            # We need the max projection for each vertex?
            # No, for a point P to be inside, max_k(n_k . (P-C)) <= R * r_in.
            # So for each vertex v, the constraint is:
            # R >= (2/sqrt(3)) * max_k ( n_k . (v - C) )
            
            # We can compute this for all vertices at once.
            # max_proj_per_vertex = np.max(projections, axis=1)
            # required_r_per_vertex = (2/sqrt3) * max_proj_per_vertex
            # required_r_for_hex = np.max(required_r_per_vertex)
            
            # Optimization: 
            # max_k ( n_k . (v - C) ) = max_k ( n_k . v - n_k . C )
            # Since C is constant for all v, we can just compute max_k ( n_k . v ) and subtract n_k . C?
            # No, max(A - B) != max(A) - B if B is a vector?
            # Here B is a scalar for each k? No.
            # n_k . (v - C) = n_k . v - n_k . C.
            # We want max_k ( n_k . v - n_k . C ).
            # This is not simply max_k(n_k . v) - max_k(n_k . C).
            
            # So we must compute for each k.
            # projections shape: (6 vertices, 6 normals)
            # We want max over normals (axis 1) for each vertex.
            max_proj_per_vertex = np.max(projections, axis=1)
            required_r_per_vertex = (2.0 / sqrt3) * max_proj_per_vertex
            required_r_for_hex = np.max(required_r_per_vertex)
            
            if required_r_for_hex > max_required_r:
                max_required_r = required_r_for_hex
                
        return max_required_r

    # Optimization Strategy:
    # The function calc_min_side(ox, oy, theta) is convex in (ox, oy) for a fixed theta?
    # It is the max of linear functions, so yes, it's convex.
    # It is also periodic in theta.
    # We can use a grid search for theta (0 to 60 degrees, due to 6-fold symmetry) 
    # and then optimize (ox, oy) using a fast method (e.g., ternary search or small grid) 
    # or just a dense grid for all 3 variables since the space is small.
    
    # Given the small dimensionality (3 variables), a dense grid search is robust and fast.
    # Range for ox, oy: The cluster is roughly within [-2, 2] x [-2, 2].
    # The optimal center will be near the cluster center (0,0).
    # Let's search ox, oy in [-1, 1] with step 0.05.
    # Let's search theta in [0, 30] with step 0.1 (symmetry reduces 60 to 30? 
    # Actually, 6-fold symmetry means 0-60 is enough. 
    # But the cluster is not 6-fold symmetric (it's a 3x4 block).
    # So we need to search 0-60 for theta.
    
    best_side = float('inf')
    best_ox = 0.0
    best_oy = 0.0
    best_theta = 0.0
    
    # To speed up, we can first do a coarse grid, then a fine grid.
    
    def run_grid_search(theta_step, ox_step, oy_step, ox_range, oy_range, theta_range):
        nonlocal best_side, best_ox, best_oy, best_theta
        
        theta_vals = np.arange(theta_range[0], theta_range[1] + theta_step/2, theta_step)
        ox_vals = np.arange(ox_range[0], ox_range[1] + ox_step/2, ox_step)
        oy_vals = np.arange(oy_range[0], oy_range[1] + oy_step/2, oy_step)
        
        # This might be too many iterations if steps are small.
        # 60/0.1 = 600 thetas.
        # 2/0.05 = 40 ox. 40 oy.
        # 600 * 40 * 40 = 960,000 evaluations.
        # Each evaluation involves matrix multiplications for 12 hexes * 6 verts * 6 normals.
        # 72 dot products. Very fast in NumPy.
        # 960k * ~10us = ~10 seconds. Acceptable.
        
        for theta in theta_vals:
            for ox in ox_vals:
                for oy in oy_vals:
                    # Time check
                    if time.time() > deadline - 5:
                        return True # Signal to stop
                    
                    side = calc_min_side(ox, oy, theta)
                    if side < best_side:
                        best_side = side
                        best_ox = ox
                        best_oy = oy
                        best_theta = theta
        return False

    # Coarse search
    # Ranges
    # Cluster extent:
    # x: -1.5*sqrt(3) to 1.5*sqrt(3) approx?
    # Let's check raw_centers x range.
    # i=0: 0, sqrt3, 2sqrt3, 3sqrt3
    # i=2: 0+sqrt3, sqrt3+sqrt3, ... -> sqrt3, 2sqrt3, 3sqrt3, 4sqrt3
    # Min x: 0. Max x: 4sqrt3. Center x: 2sqrt3.
    # Range: [-2sqrt3, 2sqrt3] approx.
    # y: 0 to 3. Center y: 1.5. Range [-1.5, 1.5].
    
    # Let's search ox, oy in [-1, 1] for now, as the outer hexagon center 
    # likely doesn't need to shift much from the cluster center.
    
    stopped = run_grid_search(
        theta_step=1.0,
        ox_step=0.1,
        oy_step=0.1,
        ox_range=(-1.0, 1.0),
        oy_range=(-1.0, 1.0),
        theta_range=(0.0, 60.0)
    )
    
    # Fine search around the best found
    if not stopped and best_side < float('inf'):
        # Narrow down the ranges
        ox_range = (best_ox - 0.2, best_ox + 0.2)
        oy_range = (best_oy - 0.2, best_oy + 0.2)
        theta_range = (max(0.0, best_theta - 2.0), min(60.0, best_theta + 2.0))
        
        stopped = run_grid_search(
            theta_step=0.1,
            ox_step=0.01,
            oy_step=0.01,
            ox_range=ox_range,
            oy_range=oy_range,
            theta_range=theta_range
        )
        
    # Even finer search if time permits
    if not stopped and best_side < float('inf'):
        ox_range = (best_ox - 0.05, best_ox + 0.05)
        oy_range = (best_oy - 0.05, best_oy + 0.05)
        theta_range = (max(0.0, best_theta - 0.5), min(60.0, best_theta + 0.5))
        
        run_grid_search(
            theta_step=0.01,
            ox_step=0.001,
            oy_step=0.001,
            ox_range=ox_range,
            oy_range=oy_range,
            theta_range=theta_range
        )
        
    # Construct the final result
    # Inner hexagons
    inner = []
    for i in range(N):
        inner.append([inner_centers[i][0], inner_centers[i][1], 0.0])
        
    outer_center = [best_ox, best_oy]
    outer_side = best_side
    outer_angle = best_theta
    
    # Fallback if something went wrong
    if outer_side > 1e9 or not math.isfinite(outer_side):
        # Use a very large hexagon
        outer_side = 10.0
        outer_center = [0.0, 0.0]
        outer_angle = 0.0
        inner = [[0.0, 0.0, 0.0]] * N # This will be invalid, but better than crash?
        # Actually, let's just return the honeycomb with a big container
        inner = [[inner_centers[i][0], inner_centers[i][1], 0.0] for i in range(N)]

    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
