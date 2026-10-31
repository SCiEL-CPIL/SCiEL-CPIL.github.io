import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Define the 11 centers based on the lattice configuration
    # Center (0,0)
    # Ring 1 (6)
    # Ring 2 (4)
    
    # Coordinates
    sqrt3 = math.sqrt(3)
    
    centers = [
        [0.0, 0.0],
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [3*sqrt3/2, 1.5],
        [3*sqrt3/2, -1.5],
        [0.0, 3.0],
        [0.0, -3.0]
    ]
    
    # Function to calculate minimum S for a given container rotation
    def calc_min_S(theta):
        # Normals of the container hexagon
        # Vertices at theta, theta+60, ...
        # Side midpoints (normals) at theta+30, theta+90, ...
        normals = []
        for k in range(6):
            angle = theta + math.pi/6 + k * math.pi/3
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_proj = 0.0
        for cx, cy in centers:
            # Check all 6 vertices of this hexagon
            # Vertices are at cx + cos(a), cy + sin(a) for a in 0, 60, ...
            for v_k in range(6):
                v_angle = v_k * math.pi/3
                vx = cx + math.cos(v_angle)
                vy = cy + math.sin(v_angle)
                
                for nx, ny in normals:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # S * sqrt(3)/2 >= max_proj
        return max_proj * 2.0 / sqrt3

    # Optimize theta
    best_S = float('inf')
    best_theta = 0.0
    
    # Coarse search
    for i in range(60):
        theta = i * math.pi / 30.0
        S = calc_min_S(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Fine search around best
    if time.time() < deadline:
        step = math.pi / 180.0
        for i in range(20):
            theta = best_theta + (i - 10) * step
            S = calc_min_S(theta)
            if S < best_S:
                best_S = S
                best_theta = theta
                
    # Even finer
    if time.time() < deadline:
        step = math.pi / 1800.0
        for i in range(40):
            theta = best_theta + (i - 20) * step
            S = calc_min_S(theta)
            if S < best_S:
                best_S = S
                best_theta = theta

    # Construct the inner list
    # The angle in the return value is the rotation of the inner hexagons.
    # Our centers are defined in a coordinate system where the hexagons are aligned with the axes (vertices at 0, 60...).
    # The container is rotated by best_theta.
    # The problem asks for:
    # inner: (x, y, angle_degrees)
    # center: (x, y) of outer
    # side: side length of outer
    # angle: rotation of outer
    
    # The "angle" of the inner hexagons in the list is relative to the global frame.
    # In our definition, the inner hexagons have vertices at 0, 60... degrees.
    # So angle_degrees = 0.0 for all.
    
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # Outer center is (0,0)
    # Outer side is best_S
    # Outer angle is best_theta (in radians? No, task says angle_degrees for inner, but for outer it just says "angle". 
    # Usually consistent units. Let's check the seed.
    # Seed: return inner, (0.0, 0.0), 40.0, 0.0
    # The last argument is "rotation". It doesn't specify degrees or radians.
    # However, the inner angle is explicitly "angle_degrees".
    # Standard practice in these tasks: if inner is degrees, outer is likely degrees too, or radians.
    # Let's assume degrees for consistency with the explicit "angle_degrees" label on inner.
    
    outer_angle_deg = math.degrees(best_theta)
    
    return inner, (0.0, 0.0), best_S, outer_angle_deg
# EVOLVE-BLOCK-END
