import numpy as np
import time
import math

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial positions
    # 1 center, 6 ring, 5 outer
    pos = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    pos[0] = [0.0, 0.0]
    
    # Ring 1
    for i in range(1, 7):
        theta = i * np.pi / 3
        pos[i] = [np.sqrt(3) * np.cos(theta), np.sqrt(3) * np.sin(theta)]
        
    # Ring 2 (5 pieces)
    for i in range(7, 12):
        theta = (i - 7) * np.pi / 3
        pos[i] = [2 * np.sqrt(3) * np.cos(theta), 2 * np.sqrt(3) * np.sin(theta)]
        
    # Pre-compute unit vectors for hexagon axes (0, 60, 120)
    # For a hexagon with side S, aligned with x-axis (flat top/bottom? or pointy?)
    # Let's assume pointy top (vertex at 90).
    # The 3 axes of symmetry are at 30, 90, 150? No.
    # The normals to the flat sides are at 0, 60, 120?
    # If pointy top, flat sides are at angles 30, 90, 150, 210, 270, 330.
    # Normals are at 0, 60, 120, 180, 240, 300.
    # So we project onto 0, 60, 120.
    # Max projection P_max.
    # Distance from center to flat side is S * cos(30) = S * sqrt(3)/2.
    # So S * sqrt(3)/2 >= P_max => S >= P_max * 2 / sqrt(3).
    
    u0 = np.array([1.0, 0.0])
    u1 = np.array([0.5, np.sqrt(3)/2])
    u2 = np.array([-0.5, np.sqrt(3)/2])
    us = [u0, u1, u2]
    
    def get_vertices(p, angle):
        # Returns 6 vertices of hexagon centered at p, rotated by angle
        # We assume side 1.
        # Vertices are at distance 1 from center.
        # Base angles: 0, 60, 120, 180, 240, 300?
        # If angle=0, vertex at (1,0).
        verts = []
        for k in range(6):
            theta = angle + k * np.pi / 3
            verts.append(p[0] + np.cos(theta), p[1] + np.sin(theta))
        return np.array(verts)
        
    def calc_side_length(positions):
        # Calculate min S for container hexagon aligned with axes 0, 60, 120
        # centered at origin (we will center the cluster first)
        # Actually, we can just compute max projection of all vertices.
        # But we need to center the cluster?
        # The container can move.
        # The min bounding hexagon size is independent of the container's position,
        # but depends on its orientation.
        # If we fix orientation, we just need the max projection of the point set
        # onto the 3 axes, relative to the centroid?
        # No, the width of the set along axis u is max(p.u) - min(p.u).
        # The container must cover this width.
        # Width W_u.
        # Container width along u is S * sqrt(3).
        # So S * sqrt(3) >= W_u => S >= W_u / sqrt(3).
        # Wait, earlier I said S >= P_max * 2 / sqrt(3).
        # If centered at origin, max(p.u) = W_u / 2.
        # So S >= (W_u / 2) * 2 / sqrt(3) = W_u / sqrt(3).
        # So S = max_u (Width_u) / sqrt(3).
        
        # We need to compute the width of the union of all hexagons along the 3 axes.
        # This is equivalent to max_proj - min_proj of all vertices.
        
        max_proj = -np.inf
        min_proj = np.inf
        
        for p in positions:
            # Since angles are 0, we can use pre-computed offsets?
            # Vertices of a unit hexagon at (0,0) angle 0:
            # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            # Projections on u0=(1,0): 1, 0.5, -0.5, -1, -0.5, 0.5. Max 1, Min -1.
            # Projections on u1=(0.5, sqrt(3)/2):
            # v0: 0.5
            # v1: 0.25 + 3/4 = 1
            # v2: -0.25 + 3/4 = 0.5
            # v3: -0.5
            # v4: -0.25 - 3/4 = -1
            # v5: 0.25 - 3/4 = -0.5
            # Max 1, Min -1.
            # It seems the projection range is always [-1, 1] for any axis?
            # No, for u0, range is 2. For u1, range is 2.
            # So the width contributed by a single hexagon is 2.
            # But we need the width of the SET.
            # So we just need max and min of (p . u + v . u) for all vertices v.
            
            # Since v . u is in [-1, 1], we can just compute p . u and add/subtract 1?
            # No, the max vertex projection depends on the direction.
            # For a unit hexagon, the support function h(u) = max(v . u) is 1 for any unit u?
            # No. The circumradius is 1. So max(v . u) is 1.
            # The inradius is sqrt(3)/2. So min(v . u) is -1?
            # Yes, for a regular hexagon, the projection on any axis has range 2.
            # So for a hexagon at p, the max projection on u is p . u + 1.
            # The min projection on u is p . u - 1.
            
            proj = np.dot(p, u0)
            max_proj = max(max_proj, proj + 1.0)
            min_proj = min(min_proj, proj - 1.0)
            
            proj = np.dot(p, u1)
            max_proj = max(max_proj, proj + 1.0)
            min_proj = min(min_proj, proj - 1.0)
            
            proj = np.dot(p, u2)
            max_proj = max(max_proj, proj + 1.0)
            min_proj = min(min_proj, proj - 1.0)
            
        width0 = max_proj - min_proj # This is mixed, we need width per axis!
        
        # Recalculate properly
        max0 = -np.inf; min0 = np.inf
        max1 = -np.inf; min1 = np.inf
        max2 = -np.inf; min2 = np.inf
        
        for p in positions:
            p0 = np.dot(p, u0)
            max0 = max(max0, p0 + 1.0)
            min0 = min(min0, p0 - 1.0)
            
            p1 = np.dot(p, u1)
            max1 = max(max1, p1 + 1.0)
            min1 = min(min1, p1 - 1.0)
            
            p2 = np.dot(p, u2)
            max2 = max(max2, p2 + 1.0)
            min2 = min(min2, p2 - 1.0)
            
        w0 = max0 - min0
        w1 = max1 - min1
        w2 = max2 - min2
        
        max_w = max(w0, w1, w2)
        S = max_w / np.sqrt(3)
        return S

    # Initial S
    current_S = calc_side_length(pos)
    best_S = current_S
    best_pos = pos.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick random hex
        i = np.random.randint(N)
        
        # Random move
        dx = np.random.normal(0, 0.5)
        dy = np.random.normal(0, 0.5)
        
        new_pos = pos.copy()
        new_pos[i] += [dx, dy]
        
        # Check collision
        valid = True
        pi = new_pos[i]
        for j in range(N):
            if i == j: continue
            pj = new_pos[j]
            dist_sq = np.sum((pi - pj)**2)
            # Min dist is sqrt(3). dist_sq >= 3
            if dist_sq < 3.0 - 1e-6:
                valid = False
                break
                
        if not valid:
            continue
            
        new_S = calc_side_length(new_pos)
        
        if new_S < current_S:
            pos = new_pos
            current_S = new_S
            if current_S < best_S:
                best_S = current_S
                best_pos = pos.copy()
        else:
            delta = new_S - current_S
            if np.random.rand() < np.exp(-delta / T):
                pos = new_pos
                current_S = new_S
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Return best
    # Center the best cluster at (0,0) for the return?
    # The task asks for inner rows (x, y, angle), outer center, side, angle.
    # We can return the positions as is, and outer center at (0,0).
    # But we should center the cluster to minimize translation?
    # The outer hexagon center can be anything.
    # Let's center the best_pos at (0,0).
    centroid = np.mean(best_pos, axis=0)
    final_pos = best_pos - centroid
    final_angles = np.zeros(N)
    
    # Outer hexagon
    outer_center = [0.0, 0.0]
    outer_side = best_S
    outer_angle = 0.0
    
    inner_list = []
    for i in range(N):
        inner_list.append([final_pos[i][0], final_pos[i][1], final_angles[i]])
        
    return inner_list, outer_center, outer_side, outer_angle
