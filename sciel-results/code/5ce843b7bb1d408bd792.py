"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    sqrt3 = np.sqrt(3.0)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Define several candidate arrangements of 11 hexagons in honeycomb lattice
    arrangements = [
        # Arrangement 1: Center + ring of 6 + 4 extending
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (2,-1), (1,1), (0,2)],
        # Arrangement 2: Different extension
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (2,-1), (1,1), (1,-2)],
        # Arrangement 3: Linear extension
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (3,0), (2,-1), (3,-1)],
        # Arrangement 4: L-shape
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (1,1), (0,2), (-1,2)],
        # Arrangement 5: Another variant
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (2,1), (1,2), (0,2)],
        # Arrangement 6: Extend in two directions
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (-2,1), (2,-1), (-1,-1)],
        # Arrangement 7: More compact
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (1,1), (2,-1), (0,2)],
    ]
    
    best_side = float('inf')
    best_arrangement = None
    best_center = (0.0, 0.0)
    best_rotation = 0.0
    
    # Unit hexagon vertices (flat-top, angle=0)
    hex_vertices = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2.0],
        [-0.5, sqrt3/2.0],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2.0],
        [0.5, -sqrt3/2.0]
    ])
    
    for arr in arrangements:
        if time.time() > deadline - 5.0:
            break
        
        # Convert to Cartesian coordinates
        centers = np.array([axial_to_cartesian(q, r) for q, r in arr])
        
        # All vertices of all inner hexagons
        all_vertices = np.vstack([hex_vertices + c for c in centers])
        
        # Centroid of cluster
        centroid = np.mean(centers, axis=0)
        
        # Try various candidate centers
        candidate_centers = [centroid]
        candidate_centers.extend(centers.tolist())
        for dx in [-1.0, 0.0, 1.0]:
            for dy in [-1.0, 0.0, 1.0]:
                candidate_centers.append([centroid[0] + dx, centroid[1] + dy])
        
        # Try various rotations of the outer hexagon
        rotations = np.linspace(0, np.pi/3, 13)
        
        for cx, cy in candidate_centers:
            if time.time() > deadline - 2.0:
                break
            center = np.array([cx, cy])
            
            for rot in rotations:
                # Edge normals for a regular hexagon with rotation rot
                # If vertex at angle rot, edge normals at rot + pi/6 + k*pi/3
                normal_angles = rot + np.pi/6 + np.arange(6) * np.pi/3
                normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
                
                # Project all vertices onto each normal
                rel_points = all_vertices - center
                projections = rel_points @ normals.T
                max_proj = np.max(projections, axis=0)
                apothem_needed = np.max(max_proj)
                
                if apothem_needed < 0:
                    apothem_needed = 0
                
                side_length = apothem_needed / (sqrt3 / 2.0)
                
                if side_length < best_side:
                    best_side = side_length
                    best_arrangement = arr
                    best_center = (float(cx), float(cy))
                    best_rotation = float(rot)
    
    if best_arrangement is not None:
        inner = []
        for q, r in best_arrangement:
            x, y = axial_to_cartesian(q, r)
            inner.append([x, y, 0.0])
        return inner, best_center, best_side, best_rotation
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END

