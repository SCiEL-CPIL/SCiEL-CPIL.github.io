import numpy as np
from itertools import combinations

def hexagon_support(direction_angle):
    verts = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)])
    d = np.array([np.cos(direction_angle), np.sin(direction_angle)])
    return np.max(verts @ d)

def calc_min_side(centers, rot_deg):
    rot_rad = np.deg2rad(rot_deg)
    normals_angles = [rot_rad + np.deg2rad(a) for a in [30, 90, 150, 210, 270, 330]]
    max_s = 0.0
    for cx, cy in centers:
        c = np.array([cx, cy])
        for na in normals_angles:
            n = np.array([np.cos(na), np.sin(na)])
            proj = c @ n
            h_in = hexagon_support(na)
            s_req = (proj + h_in) / (np.sqrt(3) / 2)
            if s_req > max_s:
                max_s = s_req
    return max_s

def construct():
    base_centers = [(0.0, 0.0)]
    for q, r in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        base_centers.append((x, y))
    
    shell2_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            s = -q - r
            if abs(q) + abs(r) + abs(s) == 4:
                x = 1.5 * q
                y = np.sqrt(3) * (r + q/2)
                shell2_points.append((x, y))
                
    best_s = float('inf')
    best_centers = None
    best_rot = 0.0
    
    for combo in combinations(shell2_points, 4):
        centers = base_centers + list(combo)
        for rot in [0, 30, 60, 90, 120, 150]:
            s = calc_min_side(centers, rot)
            if s < best_s:
                best_s = s
                best_centers = centers
                best_rot = rot

    inner = [[c[0], c[1], 0.0] for c in best_centers]
    return inner, (0.0, 0.0), best_s, best_rot

# EVOLVE-BLOCK-END
