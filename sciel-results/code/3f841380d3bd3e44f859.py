"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # We will use a hexagonal grid packing.
    # To ensure no overlap due to floating point precision or evaluator strictness,
    # we will scale the positions slightly.
    
    # Generate positions for a hexagonal cluster large enough to contain N points
    # Radius 0: 1
    # Radius 1: 7
    # Radius 2: 19
    # So for N=11, Radius 2 is sufficient.
    
    max_radius = 0
    count = 1
    while count < N:
        max_radius += 1
        # Number of points in hex cluster of radius R is 1 + 3*R*(R+1)
        count = 1 + 3 * max_radius * (max_radius + 1)
        
    positions = []
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            if abs(q + r) <= max_radius:
                # Flat-top hexagon axial to cartesian
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                positions.append((x, y))
                
    # Sort by distance from center to pick the most compact N points
    positions_sorted = sorted(positions, key=lambda p: p[0]**2 + p[1]**2)
    selected = positions_sorted[:N]
    
    # Scale positions to ensure strict non-overlap
    # The minimum distance is sqrt(3). We scale by 1.001 to be safe.
    scale = 1.001
    inner = [[x * scale, y * scale, 0.0] for x, y in selected]
    
    # Calculate the minimal enclosing regular hexagon (aligned, angle 0)
    # The inner hexagons are aligned with the outer one.
    # The "safe zone" for the centers is a hexagon of side S_safe.
    # The outer hexagon side S = S_safe + 1.
    
    # Normals for flat-top hexagon (faces perpendicular to these angles)
    # Angles: 30, 90, 150, 210, 270, 330
    angles = [math.radians(a) for a in [30, 90, 150, 210, 270, 330]]
    
    max_proj = 0.0
    for x, y, _ in inner:
        for a in angles:
            proj = x * math.cos(a) + y * math.sin(a)
            if proj > max_proj:
                max_proj = proj
                
    # max_proj is the maximum distance from center to the "supporting line" of the point set
    # in the direction of the hexagon face normals.
    # For a regular hexagon of side s, the distance from center to face (inradius) is s * sqrt(3) / 2.
    # So we need s_safe * sqrt(3) / 2 >= max_proj
    s_safe = max_proj * 2.0 / math.sqrt(3)
    
    # Outer side length
    side = s_safe + 1.0
    
    # Center is at origin
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is for evolutionary modifications)
# EVOLVE-BLOCK-END
