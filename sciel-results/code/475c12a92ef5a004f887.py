"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two hexagons overlap using Separating Axis Theorem"""
        axes = []
        for i in range(3):
            edge = v1[(i+1) % 6] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            axes.append(normal)
            edge = v2[(i+1) % 6] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
        return True
    
    # Helper: compute minimum outer side length needed to contain all inner hexagons
    def compute_min_side(inner_hexes, outer_center, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons"""
        outer_center = np.array(outer_center)
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(6):
            normal_angle = outer_angle_rad + i * math.pi / 3
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            for cx, cy, angle_deg in inner_hexes:
                verts = hex_vertices(cx, cy, angle_deg)
                rel_verts = verts - outer_center
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)
    
    # Helper: check if all hexagons are inside the outer hexagon
    def is_inside_outer(inner_hexes, outer_center, outer_side, outer_angle_deg):
        outer_center = np.array(outer_center)
        outer_angle_rad = math.radians(outer_angle_deg)
        apothem = outer_side * math.sqrt(3) / 2.0
        
        for i in range(6):
            normal_angle = outer_angle_rad + i * math.pi / 3
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            for cx, cy, angle_deg in inner_hexes:
                verts = hex_vertices(cx, cy, angle_deg)
                rel_verts = verts - outer_center
                proj = np.dot(rel_verts, normal)
                if np.max(proj) > apothem + 1e-6:
                    return False
        return True
    
    # Build discrete hexagonal lattice topologies
    # Hexagons tile the plane; use axial coordinates (q, r) for a hex grid
    # Two hexagons with axial coords (q1,r1) and (q2,r2) are adjacent if
    # hex distance = 1. The center-to-center distance for adjacent hexagons
    # (side 1) is sqrt(3) * 1 = sqrt(3).
    # 
    # For a hex grid with "flat-top" orientation (angle 0):
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    # Adjacent hexagons are at distance sqrt(3) apart.
    
    # Generate candidate topologies as sets of 12 axial coordinates
    # that form connected clusters (each hex within hex-distance 1 of at least
    # one other, and the whole set is connected).
    
    def hex_distance(q1, r1, q2, r2):
        """Hex grid distance in axial coordinates"""
        dq = q1 - q2
        dr = r1 - r2
        ds = -dq - dr
        return (abs(dq) + abs(dr) + abs(ds)) // 2
    
    def axial_to_xy(q, r, scale=1.0):
        """Convert axial coords to Cartesian for flat-top hexagons of side 'scale'"""
        x = math.sqrt(3) * scale * (q + r / 2.0)
        y = 1.5 * scale * r
        return x, y
    
    def generate_connected_clusters(n_hex, max_radius=3):
        """Generate connected sets of n_hex axial coordinates"""
        clusters = set()
        
        # BFS/DFS from origin to find all connected subsets of size n_hex
        # within a bounded region
        from itertools import product
        
        # All possible axial positions within max_radius
        candidates = []
        for q in range(-max_radius, max_radius + 1):
            for r in range(-max_radius, max_radius + 1):
                if hex_distance(0, 0, q, r) <= max_radius:
                    candidates.append((q, r))
        
        # Use a simple enumeration: start with (0,0), add neighbors
        # This is expensive for large n, but n=12 and max_radius=3 is manageable
        # We'll use a different approach: enumerate all 12-subsets of a small region
        # and check connectivity.
        
        # Actually, let's be smarter. Generate clusters by growing from (0,0).
        # For efficiency, we'll limit to a few well-known topologies.
        
        return list(clusters)
    
    # Predefined topologies (axial coordinates) for 12 hexagons
    # These are known good packings for hexagonal lattices
    
    topologies = []
    
    # Topology 1: 3x4 rectangular grid (in hex grid terms, this is a parallelogram)
    # In axial coords, a "rectangle" of width 3 and height 4:
    # q: 0..2, r: 0..3 with appropriate offset
    # Actually for flat-top hexagons, a 3-wide by 4-tall block:
    for r in range(4):
        for q in range(3):
            topologies.append((q, r))
    topologies_set = set(topologies)
    
    # Let me define several distinct topologies properly
    
    topo_list = []
    
    # T1: 3x4 block (parallelogram in hex grid)
    t1 = set()
    for r in range(4):
        for q in range(3):
            t1.add((q, r))
    topo_list.append(t1)
    
    # T2: 4x3 block (transposed)
    t2 = set()
    for r in range(3):
        for q in range(4):
            t2.add((q, r))
    topo_list.append(t2)
    
    # T3: Central hex + 6 neighbors + 5 more from next ring
    # Ring 0: (0,0)
    # Ring 1: 6 hexagons at distance 1
    # Ring 2: pick 5 from the 12 at distance 2
    t3 = set()
    t3.add((0, 0))
    ring1 = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    for c in ring1:
        t3.add(c)
    # Pick 5 from ring 2 to make a nice shape
    # Ring 2 positions (distance 2 from origin):
    ring2 = [(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1),(-1,0),(-2,0),(-2,1),(-2,2),(-1,2),(0,2)]
    # Pick to form a roughly circular shape
    picks = [(2,0),(2,-1),(0,-2),(-2,0),(-2,1)]
    for c in picks:
        t3.add(c)
    topo_list.append(t3)
    
    # T4: 2x6 strip
    t4 = set()
    for r in range(2):
        for q in range(6):
            t4.add((q, r))
    topo_list.append(t4)
    
    # T5: 6x2 strip
    t5 = set()
    for r in range(6):
        for q in range(2):
            t5.add((q, r))
    topo_list.append(t5)
    
    # T6: 3-wide, 4-tall but staggered (honeycomb pattern)
    t6 = set()
    # Row 0: q=0,1,2
    # Row 1: q=0,1,2  (offset by half in hex grid)
    # Row 2: q=0,1,2
    # Row 3: q=0,1,2
    # This is same as T1 in axial coords. Let me try a different shape.
    # T6: A "hexagon of hexagons" - center + 6 around + 5 more
    # Actually let me try: 5 in a row, then 4, then 3 (triangular)
    t6 = set()
    # Triangle: row 0 has 5, row 1 has 4, row 2 has 3
    for q in range(5):
        t6.add((q, 0))
    for q in range(4):
        t6.add((q, 1))
    for q in range(3):
        t6.add((q, 2))
    topo_list.append(t6)
    
    # T7: 4 in a row, 4 in next row, 4 in next row (but hex-staggered)
    t7 = set()
    for r in range(3):
        for q in range(4):
            t7.add((q, r))
    topo_list.append(t7)
    
    # T8: 2x2 blocks arranged in a 3x2 superlattice... 
    # Let me try: 4 clusters of 3 hexagons each
    # Each cluster: 3 hexagons in a line
    t8 = set()
    # Cluster 1: (0,0),(1,0),(2,0)
    # Cluster 2: (0,2),(1,2),(2,2)  
    # Cluster 3: (3,1),(4,1),(5,1)
    # Cluster 4: (3,3),(4,3),(5,3)
    for base_q, base_r in [(0,0),(0,2),(3,1),(3,3)]:
        for i in range(3):
            t8.add((base_q + i, base_r))
    topo_list.append(t8)
    
    # T9: More compact - try a 4-4-4 arrangement with hex staggering
    # In hex grid, adjacent rows are offset
    t9 = set()
    # Row 0: 4 hexagons
    for q in range(4):
        t9.add((q, 0))
    # Row 1: 4 hexagons, shifted
    for q in range(4):
        t9.add((q, 1))
    # Row 2: 4 hexagons
    for q in range(4):
        t9.add((q, 2))
    topo_list.append(t9)
    
    # T10: Cross-like shape
    t10 = set()
    # Center column: 4 hexagons
    for r in range(4):
        t10.add((0, r))
    # Left arm: 4 hexagons
    for r in range(4):
        t10.add((-1, r))
    # Right arm: 4 hexagons  
    for r in range(4):
        t10.add((1, r))
    topo_list.append(t10)
    
    # Now for each topology, convert to Cartesian, center it, and optimize
    # global orientation and translation to minimize enclosing hexagon side
    
    best_side = float('inf')
    best_result = None
    
    for topo in topo_list:
        if time.time() > deadline - 5:
            break
        
        # Convert to Cartesian coordinates
        coords = []
        for (q, r) in topo:
            x, y = axial_to_xy(q, r)
            coords.append((x, y))
        
        # Center the cluster at origin
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        
        centered_coords = [(x - cx, y - cy) for (x, y) in coords]
        
        # All hexagons have angle 0 in the lattice frame
        # We need to find the best global rotation and outer hexagon orientation
        # to minimize the enclosing hexagon side length
        
        def objective(params):
            """
            params: [inner_rotation_deg, outer_rotation_deg]
            We rotate all inner hexagons by inner_rotation_deg around the cluster center,
            then find the optimal outer hexagon orientation to minimize side length.
            """
            inner_rot = params[0]
            outer_rot = params[1]
            
            # Rotate all hexagon centers and their orientations
            inner_hexes = []
            rot_rad = math.radians(inner_rot)
            for (x, y) in centered_coords:
                # Rotate position around origin
                xr = x * math.cos(rot_rad) - y * math.sin(rot_rad)
                yr = x * math.sin(rot_rad) + y * math.cos(rot_rad)
                # Hexagon angle also rotates
                angle = inner_rot
                inner_hexes.append((xr, yr, angle))
            
            # Compute minimum enclosing hexagon side with given outer rotation
            side = compute_min_side(inner_hexes, (0.0, 0.0), outer_rot)
            return side
        
        # Optimize over inner rotation and outer rotation
        # Use multiple starting points
        best_for_topo = float('inf')
        best_params = None
        
        for init_ir in [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]:
            if time.time() > deadline - 5:
                break
            
            for init_or in [0, 15, 30, 45, 60]:
                if time.time() > deadline - 5:
                    break
                
                init_params = [float(init_ir), float(init_or)]
                
                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 2000,
                            'maxfev': 5000,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    
                    if result.fun < best_for_topo:
                        best_for_topo = result.fun
                        best_params = result.x.copy()
                except Exception:
                    pass
        
        if best_params is not None and best_for_topo < best_side:
            best_side = best_for_topo
            best_result = (topo, best_params.copy())
    
    # If no optimization succeeded, fall back to the first topology
    if best_result is None:
        # Use topology 1 (3x4 block) with no rotation
        topo = topo_list[0]
        coords = []
        for (q, r) in topo:
            x, y = axial_to_xy(q, r)
            coords.append((x, y))
        
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        
        centered_coords = [(x - cx, y - cy) for (x, y) in coords]
        
        inner_hexes = [(x, y, 0.0) for (x, y) in centered_coords]
        final_side = compute_min_side(inner_hexes, (0.0, 0.0), 0.0)
        
        inner = []
        for x, y, angle in inner_hexes:
            inner.append([float(x), float(y), float(angle)])
        
        return inner, (0.0, 0.0), float(final_side), 0.0
    
    # Extract the best result
    topo, params = best_result
    inner_rot = params[0]
    outer_rot = params[1]
    
    # Build final hexagons
    coords = []
    for (q, r) in topo:
        x, y = axial_to_xy(q, r)
        coords.append((x, y))
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    centered_coords = [(x - cx, y - cy) for (x, y) in coords]
    
    inner_hexes = []
    rot_rad = math.radians(inner_rot)
    for (x, y) in centered_coords:
        xr = x * math.cos(rot_rad) - y * math.sin(rot_rad)
        yr = x * math.sin(rot_rad) + y * math.cos(rot_rad)
        inner_hexes.append((xr, yr, inner_rot))
    
    # Verify no overlaps
    has_overlap = False
    for i in range(len(inner_hexes)):
        if time.time() > deadline - 2:
            break
        for j in range(i + 1, len(inner_hexes)):
            v1 = hex_vertices(*inner_hexes[i])
            v2 = hex_vertices(*inner_hexes[j])
            if hexagons_overlap(v1, v2):
                has_overlap = True
                break
        if has_overlap:
            break
    
    if has_overlap:
        # Fall back to unrotated lattice (shouldn't happen for valid topologies)
        inner_hexes = [(x, y, 0.0) for (x, y) in centered_coords]
        final_side = compute_min_side(inner_hexes, (0.0, 0.0), 0.0)
        outer_rot = 0.0
    else:
        final_side = compute_min_side(inner_hexes, (0.0, 0.0), outer_rot)
    
    inner = []
    for x, y, angle in inner_hexes:
        inner.append([float(x), float(y), float(angle)])
    
    return inner, (0.0, 0.0), float(final_side), float(outer_rot)
# EVOLVE-BLOCK-END

