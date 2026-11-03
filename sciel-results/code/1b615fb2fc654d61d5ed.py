import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3) # Min distance between centers of aligned unit hexagons

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0

    # Vertices of a unit hexagon (pointy-top, side 1)
    # Center (0,0). Vertices at angles 30, 90, 150, ...
    # Using pointy-top so that "flat" sides are horizontal/vertical-ish
    # Actually, let's stick to the standard: vertices at 0, 60, 120...
    # The shape is symmetric under 60 deg rotation.
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    # Define the hexagonal cluster topology
    # Center: (0,0)
    # Ring 1: 6 neighbors at distance D
    # Ring 2: We need 4 more. In a full hex cluster of 19, ring 2 has 12.
    # We pick 4 from ring 2 to form a compact shape.
    # Symmetry: We can pick 4 that are "corners" or "edges".
    # Let's try picking 4 that are symmetric.
    # Ring 2 coordinates in axial (q, r):
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,2)
    # Wait, standard axial to Cartesian:
    # x = D * (q + r/2)
    # y = D * (sqrt(3)/2) * r
    
    # Let's generate ring 2 points explicitly
    ring2_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = D * (q + r / 2.0)
                y = D * (math.sqrt(3) / 2.0) * r
                ring2_pts.append((x, y))
    
    # We have 12 points in ring 2.
    # We need to choose 4.
    # To minimize the bounding hexagon, we likely want a symmetric distribution.
    # Option A: 4 "corners" (e.g., (2,0), (0,2), (-2,0), (0,-2) in axial? No, those are ring 2?
    # (2,0) -> q=2,r=0 -> max(2,0,-2)=2. Yes.
    # (0,2) -> q=0,r=2 -> max(0,2,-2)=2. Yes.
    # (-2,0) -> q=-2,r=0. Yes.
    # (0,-2) -> q=0,r=-2. Yes.
    # These 4 are at distance 2*D from center.
    # Let's check if they are valid.
    
    # Option B: 4 "edge midpoints"?
    
    # Let's just brute force the choice of 4 from 12? 12C4 = 495.
    # That's fast.
    # Then for each choice, we have a fixed set of 11 centers.
    # We then optimize the global rotation theta of this cluster relative to the outer hexagon.
    
    best_S_global = float('inf')
    best_config_global = None
    best_theta_global = 0.0
    best_ring2_indices = None

    # Precompute ring 1 points (6)
    ring1_pts = []
    for i in range(6):
        a = i * math.pi / 3
        ring1_pts.append((D * math.cos(a), D * math.sin(a)))
    
    center_pt = (0.0, 0.0)
    
    # Function to calculate S for a given set of centers and theta
    def calc_S(theta, centers):
        # centers: (11, 2)
        # Rotate centers by theta? Or rotate the outer hexagon?
        # It's equivalent to rotating the inner cluster by -theta relative to fixed outer hexagon.
        # Or fixing inner cluster and rotating outer.
        # Let's fix the inner cluster as defined (aligned with x-axis mostly) and rotate the OUTER hexagon.
        # Outer hexagon normals are at angles: theta, theta+60, ...
        
        angles = theta + np.arange(6) * (math.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
        
        # Vertices of all hexagons
        # centers: (11, 2)
        # hex_verts: (6, 2)
        # We want (11*6, 2)
        all_verts = (centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]).reshape(-1, 2)
        
        # Projections
        # The width of the inner cluster in direction n is max(dot(v, n)) - min(dot(v, n))
        # But the outer hexagon is centered at (0,0).
        # For a regular hexagon of side S centered at origin, the distance from center to side is S * sqrt(3)/2.
        # The condition for a point p to be inside is |dot(p, n_i)| <= S * sqrt(3)/2 for all 6 normals n_i.
        # So S >= max_i ( max_j |dot(v_j, n_i)| ) / (sqrt(3)/2)
        
        projs = np.abs(all_verts @ normals.T) # (66, 6)
        max_proj = np.max(projs)
        return (2.0 / math.sqrt(3)) * max_proj

    # Iterate over choices of 4 from 12 ring-2 points
    num_ring2 = len(ring2_pts)
    for combo in itertools.combinations(range(num_ring2), 4):
        if time.time() > deadline - safety_margin:
            break
            
        # Construct centers
        # 1 center + 6 ring1 + 4 ring2
        centers_list = [center_pt] + ring1_pts + [ring2_pts[i] for i in combo]
        centers = np.array(centers_list)
        
        # Check if this configuration is valid (no overlaps)
        # Since it's a subset of the perfect hex grid, and we only picked non-adjacent ring2 points?
        # Wait, ring2 points are distance 2*D from center, and distance D from ring1 points?
        # Distance between two ring2 points can be D (if adjacent in ring2).
        # If we pick two adjacent ring2 points, they are distance D apart.
        # Hexagons at distance D touch. So they are valid (disjoint interiors).
        # So any subset of the hex grid points is a valid packing (touching is allowed, overlapping is not).
        # Since our points are from the hex grid, the distance between any two is >= D.
        # So validity is guaranteed.
        
        # Optimize theta
        # Coarse scan
        coarse_thetas = np.linspace(0, math.pi / 6, 60) # Symmetry of hexagon is 60 deg, but cluster might have lower symmetry.
        # Actually, the outer hexagon has 60 deg symmetry. The inner cluster might have 60, 120, 180...
        # Let's scan 0 to 60 deg.
        coarse_Ss = np.array([calc_S(t, centers) for t in coarse_thetas])
        min_idx = np.argmin(coarse_Ss)
        
        # Refine
        try:
            res = minimize_scalar(lambda t: calc_S(t, centers), bounds=(coarse_thetas[max(0, min_idx-1)], coarse_thetas[min(59, min_idx+1)]), method='bounded', options={'xatol': 1e-6})
            if res.success:
                S_val = res.fun
                theta_val = res.x
            else:
                S_val = coarse_Ss[min_idx]
                theta_val = coarse_thetas[min_idx]
        except:
            S_val = coarse_Ss[min_idx]
            theta_val = coarse_thetas[min_idx]

        if S_val < best_S_global:
            best_S_global = S_val
            best_config_global = centers.copy()
            best_theta_global = theta_val
            best_ring2_indices = combo

    # Fallback
    if best_config_global is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner_rows = []
    for cx, cy in best_config_global:
        inner_rows.append([float(cx), float(cy), 0.0])
        
    center = (0.0, 0.0)
    side = float(best_S_global)
    angle_deg = float(math.degrees(best_theta_global))
    
    return inner_rows, center, side, angle_deg
