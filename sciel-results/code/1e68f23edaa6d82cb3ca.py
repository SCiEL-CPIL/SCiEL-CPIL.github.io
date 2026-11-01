import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Lattice vectors for hexagonal packing
    # Hexagons of side 1 touch at distance sqrt(3)
    e1 = np.array([np.sqrt(3), 0])
    e2 = np.array([np.sqrt(3)/2, 3/2])

    # Generate candidate configurations of 12 lattice points
    candidates = []

    # Candidate 1: 3x4 block
    pts_34 = []
    for i in range(3):
        for j in range(4):
            pts_34.append(i * e1 + j * e2)
    candidates.append(np.array(pts_34))

    # Candidate 2: 2x6 block
    pts_26 = []
    for i in range(2):
        for j in range(6):
            pts_26.append(i * e1 + j * e2)
    candidates.append(np.array(pts_26))

    # Candidate 3: 4x3 block (transposed)
    pts_43 = []
    for i in range(4):
        for j in range(3):
            pts_43.append(i * e1 + j * e2)
    candidates.append(np.array(pts_43))

    # Candidate 4: "Flower" (7) + 5 points from ring 2
    # Center
    flower_pts = [np.array([0.0, 0.0])]
    # Ring 1 (6 points)
    ring1_coords = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for i, j in ring1_coords:
        flower_pts.append(i * e1 + j * e2)
        
    # Ring 2 candidates (12 points)
    # We want to pick 5 that minimize the bounding box
    # Let's try picking 5 that are "evenly" distributed or form a cap
    # Option A: Cap (all on one side)
    ring2_cap = [(2,0), (1,1), (0,2), (-1,2), (-2,1)]
    for i, j in ring2_cap:
        flower_pts.append(i * e1 + j * e2)
    candidates.append(np.array(flower_pts))

    # Option B: Spread out (every other one)
    flower_pts2 = [np.array([0.0, 0.0])]
    for i, j in ring1_coords:
        flower_pts2.append(i * e1 + j * e2)
    ring2_spread = [(2,0), (0,2), (-2,0), (0,-2), (1,-2)] # Arbitrary selection
    for i, j in ring2_spread:
        flower_pts2.append(i * e1 + j * e2)
    candidates.append(np.array(flower_pts2))

    # Function to compute min side length for a given cluster and rotation
    def compute_side_for_cluster(cluster, theta):
        # Rotate cluster
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_cluster = cluster @ R.T
        
        # Centroid
        centroid = np.mean(rotated_cluster, axis=0)
        centered_cluster = rotated_cluster - centroid
        
        # Get vertices of all hexagons
        # Unit hexagon vertices (angle 0: vertex at 0 deg)
        angles = np.arange(6) * np.pi / 3
        unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Outer hexagon normals (fixed)
        normal_angles = angles + np.pi / 6
        unit_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        max_extent = 0.0
        for center in centered_cluster:
            verts = unit_verts + center
            # Project onto normals
            proj = verts @ unit_normals.T
            extents = np.max(np.abs(proj), axis=0)
            local_max = np.max(extents)
            if local_max > max_extent:
                max_extent = local_max
        
        # Apothem a = max_extent
        # Side S = 2 * a / sqrt(3)
        return 2.0 * max_extent / np.sqrt(3)

    best_params = None
    best_score = np.inf
    best_cluster = None

    for cluster in candidates:
        if time.time() > deadline:
            break
            
        # Optimize rotation theta
        # Search 0 to pi/3
        res = minimize_scalar(compute_side_for_cluster, bounds=(0, np.pi/3), args=(cluster,), method='bounded', options={'xatol': 1e-6})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            best_cluster = cluster

    # Fallback
    if best_cluster is None:
        best_cluster = candidates[0]
        best_params = 0.0
        best_score = compute_side_for_cluster(best_cluster, 0.0)

    # Finalize positions
    c, s = np.cos(best_params), np.sin(best_params)
    R = np.array([[c, -s], [s, c]])
    rotated_cluster = best_cluster @ R.T
    centroid = np.mean(rotated_cluster, axis=0)
    centered_cluster = rotated_cluster - centroid

    inner = []
    for center in centered_cluster:
        x, y = center
        inner.append([float(x), float(y), float(np.degrees(best_params))])
    
    center = (0.0, 0.0)
    side = float(best_score)
    angle = 0.0
    
    return inner, center, side, angle
