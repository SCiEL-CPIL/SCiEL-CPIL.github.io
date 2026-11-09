"""Packing 11 unit hexagons into smallest regular hexagon."""
import numpy as np
import math
import time
import random

N = 11

def _hex_vertices(cx, cy, angle_deg):
    """Get 6 vertices of unit hexagon centered at (cx,cy) with rotation in degrees."""
    theta = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = theta + k * math.pi / 3.0
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_outer_side(centers, angles):
    """Compute minimum outer hexagon side (circumradius) for given placements."""
    sqrt3 = math.sqrt(3.0)
    max_r = 0.0
    for (cx, cy), ang in zip(centers, angles):
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            r1 = abs(vy)
            r2 = abs(sqrt3 * vx + vy) / sqrt3
            r3 = abs(sqrt3 * vx - vy) / sqrt3
            r = max(r1, r2, r3)
            if r > max_r:
                max_r = r
    return max_r

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    sqrt3 = math.sqrt(3.0)
    
    # Hexagonal grid layout: center + ring1(6) + 4 from ring2 = 11
    # Axial coords (pointy-top hex grid)
    positions_axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (1, 1), (-1, -1),
    ]
    
    # Convert axial to cartesian (pointy-top)
    centers = []
    for (q, r) in positions_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    angles = [0.0] * N  # pointy-top orientation
    
    # Compute initial side
    best_side = _min_outer_side(centers, angles)
    best_angles = angles[:]
    best_centers = centers[:]
    
    # Optimize translation offset
    best_offset = (0.0, 0.0)
    for dx in np.linspace(-0.5, 0.5, 21):
        for dy in np.linspace(-0.5, 0.5, 21):
            shifted = [(x + dx, y + dy) for (x, y) in centers]
            s = _min_outer_side(shifted, angles)
            if s < best_side:
                best_side = s
                best_offset = (dx, dy)
    
    dx, dy = best_offset
    best_centers = [(x + dx, y + dy) for (x, y) in centers]
    
    # Try flat-top orientation for all
    angles_flat = [30.0] * N
    s_flat = _min_outer_side(best_centers, angles_flat)
    if s_flat < best_side:
        best_side = s_flat
        best_angles = angles_flat[:]
    
    # Try random rotations if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        random.seed(42)
        candidate_angles = [0.0, 30.0, 60.0, 15.0, 45.0, 75.0, 90.0]
        for trial in range(500):
            if time.time() > deadline - 5:
                break
            test_angles = [random.choice(candidate_angles) for _ in range(N)]
            s = _min_outer_side(best_centers, test_angles)
            if s < best_side:
                best_side = s
                best_angles = test_angles[:]
        
        # Also try small position perturbations
        if time.time() < deadline - 10:
            for trial in range(200):
                if time.time() > deadline - 5:
                    break
                perturbed = []
                for (x, y) in best_centers:
                    px = x + random.uniform(-0.1, 0.1)
                    py = y + random.uniform(-0.1, 0.1)
                    perturbed.append((px, py))
                s = _min_outer_side(perturbed, best_angles)
                if s < best_side:
                    best_side = s
                    best_centers = perturbed
    
    inner = [[x, y, a] for (x, y), a in zip(best_centers, best_angles)]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
