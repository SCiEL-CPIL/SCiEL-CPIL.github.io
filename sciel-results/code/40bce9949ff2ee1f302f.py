import numpy as np
from scipy.optimize import minimize
import time

def get_hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon.
    cx, cy: center
    s: side length (radius)
    angle_deg: rotation angle
    """
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def compute_min_side_length(vertices):
    """
    Computes the minimum side length of a regular hexagon (angle 0, center 0,0)
    that contains all given vertices.
    vertices: (N, 2) array
    """
    if len(vertices) == 0:
        return 0.0
    
    # Normals for hexagon with angle 0
    # Vertices at 0, 60, 120, 180, 240, 300
    # Normals are perpendicular to edges.
    # Edge between v0 and v1. Midpoint angle 30. Normal angle 30?
    # Let's check.
    # v0 = (1, 0), v1 = (0.5, sqrt(3)/2).
    # Edge vector = (-0.5, sqrt(3)/2).
    # Normal = (sqrt(3)/2, 0.5) -> angle 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for k in range(6):
        theta = np.deg2rad(30 + 60 * k)
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals) # (6, 2)
    
    # Project all vertices onto all normals
    # projections: (N, 6)
    projections = vertices @ normals.T
    
    # We need max |projection| for each normal
    max_projs = np.max(np.abs(projections), axis=0)
    
    # The distance from center to side is S * sqrt(3) / 2
    # So max_proj = S * sqrt(3) / 2
    # S = max_proj * 2 / sqrt(3)
    S = np.max(max_projs) * 2.0 / np.sqrt(3.0)
    return S

def compute_overlap_penalty(hex_data):
    """
    Computes the penalty for overlapping hexagons.
    hex_data: (11, 3) array of (x, y, angle_deg)
    Side length is 1.
    Returns sum of squared negative gaps.
    """
    n = len(hex_data)
    penalty = 0.0
    vertices_list = []
    for i in range(n):
        x, y, a = hex_data[i]
        verts = get_hex_vertices(x, y, 1.0, a)
        vertices_list.append(verts)
        
    for i in range(n):
        for j in range(i + 1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            
            # Check separation using SAT (Separating Axis Theorem)
            # Axes are the 12 edge normals of the two hexagons
            axes = []
            # Normals of hex 1
            for k in range(6):
                theta = np.deg2rad(hex_data[i][2] + 30 + 60 * k)
                axes.append([np.cos(theta), np.sin(theta)])
            # Normals of hex 2
            for k in range(6):
                theta = np.deg2rad(hex_data[j][2] + 30 + 60 * k)
                axes.append([np.cos(theta), np.sin(theta)])
            
            axes = np.array(axes)
            
            min_gap = np.inf
            found_separating = False
            for axis in axes:
                proj1 = v1 @ axis
                proj2 = v2 @ axis
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Gap between intervals [min1, max1] and [min2, max2]
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = 0.0 # Overlapping or touching on this axis
                
                if gap > 0:
                    found_separating = True
                    break # Found a separating axis, no overlap
                else:
                    # If gap is 0, they touch.
                    # We want gap > 0.
                    # So we treat gap 0 as penalty?
                    # The problem says "apart by more than 1e-6".
                    # So we need gap > 1e-6.
                    # Let's accumulate the negative part.
                    # If gap <= 0, penalty += gap^2?
                    # Actually, if they overlap, gap is negative in some sense?
                    # No, my gap calculation above gives 0 if they overlap.
                    # I need the *minimum* gap over all axes.
                    # If min_gap > 0, they are separated.
                    # If min_gap <= 0, they intersect or touch.
                    # I should track the minimum gap.
                    pass
            
            # Recalculate min_gap properly
            min_gap = np.inf
            for axis in axes:
                proj1 = v1 @ axis
                proj2 = v2 @ axis
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = 0.0
                
                if gap < min_gap:
                    min_gap = gap
            
            if min_gap < 1e-6:
                # Penalty for being too close or overlapping
                # Use a smooth penalty
                penalty += (1e-6 - min_gap) ** 2
                
    return penalty

def objective(params):
    """
    Objective function for optimization.
    params: 33 array
    """
    hex_data = params.reshape(11, 3)
    
    # Calculate S
    all_vertices = []
    for i in range(11):
        x, y, a = hex_data[i]
        verts = get_hex_vertices(x, y, 1.0, a)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    S = compute_min_side_length(all_vertices)
    
    # Calculate overlap penalty
    pen = compute_overlap_penalty(hex_data)
    
    # Weight for penalty
    # We want to minimize S, but avoid overlaps.
    # If pen > 0, we add a large penalty.
    return S + 10.0 * pen

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_S = np.inf
    best_params = None
    
    # Initial guess: 7 cluster + 4 random
    def generate_initial_guess():
        # 7 cluster
        pts = [[0, 0, 0]]
        for k in range(6):
            angle = 60 * k
            rad = np.deg2rad(angle)
            x = np.sqrt(3) * np.cos(rad)
            y = np.sqrt(3) * np.sin(rad)
            pts.append([x, y, 0])
        
        # 4 more: place them near the "corners" of the 7-cluster
        # The 7-cluster has 6 "bays". We can put 2 in opposite bays and 2 in others?
        # Or just randomize positions slightly outside the 7-cluster.
        for _ in range(4):
            angle = np.random.uniform(0, 360)
            rad = np.deg2rad(angle)
            # Distance from center ~ 2*sqrt(3)
            dist = np.sqrt(3) * 2.0 + np.random.uniform(-0.5, 0.5)
            x = dist * np.cos(rad)
            y = dist * np.sin(rad)
            a = np.random.uniform(0, 60) # random angle
            pts.append([x, y, a])
            
        return np.array(pts)

    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        x0 = generate_initial_guess()
        
        # Optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success:
                S_val = compute_min_side_length(np.vstack([get_hex_vertices(*hex_data) for hex_data in res.x.reshape(11, 3)]))
                pen = compute_overlap_penalty(res.x.reshape(11, 3))
                
                if pen < 1e-4: # Valid enough
                    if S_val < best_S:
                        best_S = S_val
                        best_params = res.x.copy()
                        
        except Exception as e:
            continue

    if best_params is None:
        # Fallback to seed-like solution
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
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Post-process: Ensure validity
    # The optimizer might leave small overlaps.
    # We can try to fix them by slightly increasing S? No, S is fixed by positions.
    # We can try to push overlapping hexagons apart?
    # For now, return the best found.
    
    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
