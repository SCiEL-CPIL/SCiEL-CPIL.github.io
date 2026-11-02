import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3)
    
    # Axial to Cartesian for flat-top hexagons, side=1
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # 11 hexagons in a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    centers = [axial_to_cartesian(q, r) for q, r in axial_coords]
    inner = [[x, y, 0.0] for x, y in centers]
    
    def min_hex_side(centers, theta_deg):
        theta = math.radians(theta_deg)
        max_s = 0.0
        for k in range(3):
            angle = theta + k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            projs = [x * nx + y * ny for x, y in centers]
            range_proj = max(projs) - min(projs)
            s_candidate = range_proj / sqrt3
            if s_candidate > max_s:
                max_s = s_candidate
        return max_s
    
    # Coarse search over theta
    best_s = float('inf')
    best_theta = 0.0
    for theta_deg in range(0, 180, 2):
        if time.time() > deadline:
            break
        s = min_hex_side(centers, theta_deg)
        if s < best_s:
            best_s = s
            best_theta = float(theta_deg)
    
    # Fine search
    for i in range(20):
        if time.time() > deadline:
            break
        theta_deg = best_theta - 2.0 + 4.0 * i / 19.0
        s = min_hex_side(centers, theta_deg)
        if s < best_s:
            best_s = s
            best_theta = theta_deg
    
    # Even finer
    for i in range(20):
        if time.time() > deadline:
            break
        theta_deg = best_theta - 0.2 + 0.4 * i / 19.0
        s = min_hex_side(centers, theta_deg)
        if s < best_s:
            best_s = s
            best_theta = theta_deg
    
    # Outer hexagon center at centroid of inner hexagons
    cx = sum(x for x, y in centers) / len(centers)
    cy = sum(y for x, y in centers) / len(centers)
    
    return inner, (cx, cy), best_s, best_theta
