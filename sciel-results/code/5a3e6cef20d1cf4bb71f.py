import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Define a cluster of 12 hexagons
    # Lattice coordinates (q, r) for a hex grid
    # Let's try a 3x4 block or similar
    # A better cluster: Hexagon of side 2 (7 hexes) + 5 more
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 hexes
    # Total 7.
    # Add 5 more: (2,0), (1,1), (0,2), (-1,2), (-2,1)?
    # Let's just use a 3x4 grid of hexes for simplicity, then optimize.
    # Grid:
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # This is a parallelogram.
    
    # Lattice to Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) / 2 * q + sqrt(3) * r
    
    hex_coords = []
    # 3 rows of 4
    for r in range(3):
        for q in range(4):
            # Center offset
            x = 1.5 * q
            y = (np.sqrt(3) / 2) * q + np.sqrt(3) * r
            hex_coords.append((x, y))
            
    # This cluster has 12 hexes.
    # We need to find the min bounding hexagon.
    
    # Vertices of a unit hexagon centered at (0,0) with angle 0
    # Side length 1
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    unit_hex_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Function to check if a hexagon is inside the container
    def is_inside(hex_center, hex_angle, container_center, container_side, container_angle):
        # Transform hex vertices to global coords
        # Rotate hex
        c, s = np.cos(hex_angle), np.sin(hex_angle)
        R = np.array([[c, -s], [s, c]])
        verts = (R @ unit_hex_verts.T).T + hex_center
        
        # Container is a regular hexagon
        # Check if all vertices are inside
        # Container vertices
        c_out, s_out = np.cos(container_angle), np.sin(container_angle)
        R_out = np.array([[c_out, -s_out], [s_out, c_out]])
        # Container center is (0,0) for local check? No, we shift.
        # Let's check distance from container center.
        # A point P is inside a regular hexagon of side S centered at C with angle A
        # if the distance to each of the 6 sides is <= apothem.
        # Apothem = S * sqrt(3) / 2.
        
        # Vector from container center to point
        # We can check by projecting onto the 6 normals of the container.
        
        # Normals of container (outward)
        # Angles of normals: container_angle + i * 60 deg
        normals = []
        for i in range(6):
            ang = container_angle + i * np.pi / 3
            normals.append([np.cos(ang), np.sin(ang)])
        normals = np.array(normals)
        
        # Translate verts to be relative to container center
        rel_verts = verts - container_center
        
        # Project onto normals
        # proj = rel_verts @ normals.T
        # Max projection must be <= apothem
        apothem = container_side * np.sqrt(3) / 2
        
        projs = rel_verts @ normals.T
        max_projs = np.max(projs, axis=1)
        
        return np.all(max_projs <= apothem + 1e-9)

    # Search for best S
    # Cluster centroid
    centroids = np.array(hex_coords)
    centroid = np.mean(centroids, axis=0)
    
    # Shift cluster so centroid is at (0,0)
    hex_coords = [(x - centroid[0], y - centroid[1]) for x, y in hex_coords]
    
    best_S = 10.0
    best_params = None
    
    # Grid search over container rotation and side length
    # Since area bound is 3.464, start from 3.5
    # Step size for S: 0.01
    # Step size for angle: 1 degree
    
    # To speed up, we can first find the bounding box of the cluster
    # and estimate a good S.
    
    # Let's try a few rotations
    for container_angle_deg in range(0, 60, 5):
        container_angle = np.radians(container_angle_deg)
        
        # Estimate min S for this angle
        # Project all vertices of all hexagons onto the 6 normals
        # The max projection determines the required apothem.
        
        # Get all vertices
        all_verts = []
        for x, y in hex_coords:
            # Hexagons are aligned with the cluster (angle 0)
            # Wait, the inner hexagons can rotate.
            # But for a fixed cluster, we assume they are all aligned with the cluster lattice.
            # This is a valid configuration.
            verts = unit_hex_verts + np.array([x, y])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        normals = []
        for i in range(6):
            ang = container_angle + i * np.pi / 3
            normals.append([np.cos(ang), np.sin(ang)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        
        # Required apothem is max(max_projs)
        req_apothem = np.max(max_projs)
        req_S = req_apothem * 2 / np.sqrt(3)
        
        if req_S < best_S:
            best_S = req_S
            best_params = (container_angle_deg, 0.0, 0.0) # Center at 0,0
            
    # Refine with finer steps
    if best_params:
        best_angle = best_params[0]
        # Fine search around best_angle
        for da in np.arange(-5, 5.1, 0.5):
            container_angle = np.radians(best_angle + da)
            normals = []
            for i in range(6):
                ang = container_angle + i * np.pi / 3
                normals.append([np.cos(ang), np.sin(ang)])
            normals = np.array(normals)
            
            all_verts = []
            for x, y in hex_coords:
                verts = unit_hex_verts + np.array([x, y])
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            projs = all_verts @ normals.T
            max_projs = np.max(projs, axis=0)
            req_apothem = np.max(max_projs)
            req_S = req_apothem * 2 / np.sqrt(3)
            
            if req_S < best_S:
                best_S = req_S
                best_params = (best_angle + da, 0.0, 0.0)

    # Construct the final list of inner hexagons
    # The cluster is centered at (0,0).
    # The outer hexagon is centered at (0,0).
    # Inner hexagons are at hex_coords with angle 0.
    
    inner = []
    for x, y in hex_coords:
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), best_S, 0.0

