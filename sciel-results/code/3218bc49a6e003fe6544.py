import numpy as np
import time
from scipy.optimize import minimize_scalar

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    # Hexagon vertices for unit side, angle 0
    # Vertices at 0, 60, ... 300
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Honeycomb lattice basis vectors
    # Distance between centers is sqrt(3)
    d = np.sqrt(3)
    u = np.array([d, 0])
    v = np.array([d * np.cos(np.deg2rad(60)), d * np.sin(np.deg2rad(60))])
    
    # Function to get lattice point
    def lattice_point(i, j):
        return i * u + j * v
        
    # Generate candidate topologies
    # Core: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (3,-1), (2,-2) ? 
    # Let's generate Ring 2 properly.
    # Ring 2 consists of points with max(|i|, |j|, |i+j|) = 2? No.
    # Hexagonal distance.
    
    # Simpler: Hardcode a few likely good sets of 11 points (i,j)
    # Core 7:
    core = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Ring 2 candidates (12 points)
    # Neighbors of Ring 1 points that are not in Core/Ring 1
    ring2_candidates = []
    # From (1,0): (2,0), (1,1), (2,-1)
    # From (0,1): (1,1), (0,2), (-1,2)
    # From (-1,1): (-1,2), (-2,1), (-1,0)-core
    # From (-1,0): (-2,0), (-2,1), (-1,-1)
    # From (0,-1): (-1,-1), (0,-2), (1,-2)
    # From (1,-1): (1,-2), (2,-1), (2,-2)
    
    unique_ring2 = [
        (2,0), (1,1), (2,-1),
        (0,2), (-1,2),
        (-2,1), 
        (-2,0), (-1,-1),
        (0,-2), (1,-2),
        (2,-2)
    ]
    # Remove duplicates and check against core
    seen = set(core)
    clean_ring2 = []
    for p in unique_ring2:
        if p not in seen:
            seen.add(p)
            clean_ring2.append(p)
            
    # clean_ring2 should have 12 points
    # Let's pick 4 from clean_ring2 to add to core
    
    topologies = []
    
    # Topology 1: Symmetric 4 (every other)
    # Indices 0, 2, 4, 6 in clean_ring2? 
    # clean_ring2 order:
    # 0: (2,0)
    # 1: (1,1)
    # 2: (2,-1)
    # 3: (0,2)
    # 4: (-1,2)
    # 5: (-2,1)
    # 6: (-2,0)
    # 7: (-1,-1)
    # 8: (0,-2)
    # 9: (1,-2)
    # 10: (2,-2)
    # Wait, I listed 11. Let's re-verify Ring 2.
    # A hexagon has 6 neighbors.
    # Ring 1 has 6 points.
    # Each Ring 1 point has 3 new neighbors in Ring 2 (one shared with prev, one shared with next).
    # So 6 * 2 = 12?
    # Let's just use a known set of 12 Ring 2 coordinates.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (3,-1)?? No.
    
    # Let's generate them programmatically to be safe.
    ring2 = set()
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Hex distance from (0,0)
            dist = max(abs(i), abs(j), abs(i+j))
            if dist == 2:
                ring2.add((i,j))
                
    ring2_list = list(ring2)
    # Should be 12 points
    
    # Candidate 1: 4 points symmetric
    # Pick 0, 3, 6, 9 if sorted?
    # Let's just try a few subsets of 4 from ring2_list
    # To save time, we will test:
    # 1. (2,0), (0,2), (-2,0), (0,-2) -> Diamond
    # 2. (2,0), (1,1), (-1,2), (-2,1) -> Block
    # 3. (2,0), (2,-1), (-2,0), (-1,-1) -> Asym
    
    candidate_sets = [
        [(2,0), (0,2), (-2,0), (0,-2)],
        [(2,0), (1,1), (-1,2), (-2,1)],
        [(2,0), (1,1), (0,2), (-1,2)], # Block of 4
        [(2,0), (2,-1), (1,-2), (0,-2)], # Corner block
    ]
    
    best_side = float('inf')
    best_result = None
    
    for adds in candidate_sets:
        if time.time() > deadline - 60:
            break
            
        points = core + adds
        # Ensure 11 points
        if len(points) != 11:
            continue
            
        # Convert to coordinates
        centers = np.array([lattice_point(i, j) for i, j in points])
        
        # Scale by 1.001 to ensure gap > 1e-6
        centers = centers * 1.001
        
        # Calculate vertices for all hexagons
        # vertices[i] is 6x2
        all_vertices = []
        for c in centers:
            verts = c + hex_verts
            all_vertices.append(verts)
        all_vertices = np.array(all_vertices) # (11, 6, 2)
        
        # Centroid
        centroid = np.mean(all_vertices, axis=(0, 1))
        all_vertices_centered = all_vertices - centroid
        
        def objective(phi):
            # Normals of outer hexagon
            # Outer hexagon rotated by phi
            # Normals at phi + 30, phi + 90, ...
            angles = np.deg2rad([30, 90, 150, 210, 270, 330]) + phi
            normals = np.array([
                [np.cos(a), np.sin(a)] for a in angles
            ])
            
            # Project all vertices onto normals
            # all_vertices_centered is (11, 6, 2)
            # normals is (6, 2)
            # We need max projection for each normal
            # Flatten vertices to (66, 2)
            verts_flat = all_vertices_centered.reshape(-1, 2)
            projs = verts_flat @ normals.T # (66, 6)
            max_projs = np.max(projs, axis=0) # (6,)
            
            # Side length S
            # Apothem = S * sqrt(3) / 2
            # Max proj <= Apothem
            # S = 2 / sqrt(3) * max(max_projs)
            max_proj_val = np.max(max_projs)
            S = (2.0 / np.sqrt(3.0)) * max_proj_val
            
            return S
            
        # Optimize phi in [0, pi/3]
        res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', 
                              options={'xatol': 1e-6})
        
        if res.success:
            phi_opt = res.x
            S_opt = res.fun
            
            if S_opt < best_side:
                best_side = S_opt
                # Store best configuration
                # Inner hexagons: centers, angle 0
                inner_data = np.column_stack([centers, np.zeros(11)])
                # Outer hexagon: center 0, angle -phi_opt (since we rotated cluster by phi, 
                # relative to outer, outer is at -phi. Or just return phi as outer angle 
                # and keep inner at 0? 
                # The task asks for (x, y, angle) for outer.
                # If we rotate the cluster by phi, and keep outer at 0, 
                # the cluster is at phi.
                # But we want to return the configuration that fits in the outer hexagon.
                # If we return inner centers as calculated (rotated by phi? No, we didn't rotate them in the array, 
                # we just projected them as if rotated).
                # Wait. In objective, we used `all_vertices_centered` which are NOT rotated.
                # We just changed the normals.
                # This is equivalent to rotating the outer hexagon by -phi relative to the cluster.
                # So the inner hexagons are at their original positions (rotated by 0).
                # The outer hexagon is rotated by -phi.
                # So outer angle = -phi_opt.
                outer_data = np.array([0, 0, np.rad2deg(-phi_opt)])
                best_result = (inner_data, outer_data, S_opt)
                
    if best_result is None:
        # Fallback
        inner_data = np.array(
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
        outer_data = np.array([0, 0, 0])
        return inner_data, outer_data, 8.0

    return best_result
