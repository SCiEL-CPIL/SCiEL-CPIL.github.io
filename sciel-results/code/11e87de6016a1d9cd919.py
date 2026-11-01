import numpy as np
import math

N = 11

def hexagon_vertices(center, angle_deg=0):
    cx, cy = center
    rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = rad + i * math.pi / 3
        x = cx + math.cos(theta)
        y = cy + math.sin(theta)
        verts.append((x, y))
    return verts

def min_hexagon_side_for_orientation(centers, inner_angle=0, container_angle=0):
    max_s = 0.0
    container_rad = math.radians(container_angle)
    for c in centers:
        verts = hexagon_vertices(c, inner_angle)
        for v in verts:
            x, y = v
            if x == 0 and y == 0:
                continue
            theta = math.atan2(y, x)
            dist = math.hypot(x, y)
            
            # Find nearest container vertex angle
            # Container vertices are at container_angle + k * 60
            # We want theta_rel = theta - (container_angle + k*60)
            # such that |theta_rel| <= 30
            # k = round((theta - container_rad) / (pi/3))
            k = round((theta - container_rad) / (math.pi / 3))
            theta_rel = theta - (container_rad + k * (math.pi / 3))
            
            # R(theta_rel) = S * (sqrt(3)/2) / cos(theta_rel)
            # S >= dist * cos(theta_rel) / (sqrt(3)/2)
            s_req = dist * math.cos(theta_rel) / (math.sqrt(3) / 2)
            if s_req > max_s:
                max_s = s_req
    return max_s

def axial_to_xy(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return (x, y)

def get_shape_7_plus_4():
    coords = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    coords.extend([(2,0), (2,-1), (1,-2), (0,-2)])
    return [axial_to_xy(q,r) for q,r in coords]

def get_shape_2x6():
    c = []
    for r in [0, 1]:
        for q in range(6):
            c.append((q, r))
    return [axial_to_xy(q,r) for q,r in c]

def get_shape_3x4():
    c = []
    for r in range(3):
        for q in range(4):
            c.append((q, r))
    return [axial_to_xy(q,r) for q,r in c]

def center_points(points):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return [(p[0]-cx, p[1]-cy) for p in points]

shapes = {
    "7+4": get_shape_7_plus_4(),
    "2x6": get_shape_2x6(),
    "3x4": get_shape_3x4(),
}

best_s = float('inf')
best_centers = None
best_container_angle = 0.0

for name, pts in shapes.items():
    centered = center_points(pts)
    # Try container angles 0 and 30
    for c_ang in [0, 30]:
        s = min_hexagon_side_for_orientation(centered, 0, c_ang)
        if s < best_s:
            best_s = s
            best_centers = centered
            best_container_angle = c_ang

inner = [[x, y, 0.0] for x, y in best_centers]
center = (0.0, 0.0)
side = best_s
angle = best_container_angle

def construct():
    return inner, center, side, angle
