import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Define 3x4 grid of pointy-top unit hexagons, centered at origin
    inner_positions = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (sqrt3 / 2 if row % 2 == 1 else 0)
            y = row * 1.5
            inner_positions.append((x, y))
    
    # Center the grid
    cx = sum(p[0] for p in inner_positions) / N
    cy = sum(p[1] for p in inner_positions) / N
    inner_positions = [(p[0] - cx, p[1] - cy) for p in inner_positions]
    
    def point_in_hexagon(px, py, hx, hy, side, angle_deg):
        """Check if point is inside hexagon using edge half-plane tests."""
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        inradius = side * sqrt3 / 2
        for k in range(6):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            if lx * nx + ly * ny > inradius + 1e-10:
                return False
        return True
    
    def get_inner_vertices(i):
        """Get 6 vertices of inner hexagon i (unit side, pointy-top)."""
        ix, iy = inner_positions[i]
        verts = []
        for k in range(6):
            a = math.radians(30 + k * 60)
            verts.append((ix + math.cos(a), iy + math.sin(a)))
        return verts
    
    # Precompute inner vertices
    all_inner_verts = [get_inner_vertices(i) for i in range(N)]
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        outer_cx, outer_cy, outer_side, outer_angle = params
        if outer_side <= 1.0:
            return 1e10
        # Check all inner hexagon vertices are inside outer hexagon
        for i in range(N):
            for vx, vy in all_inner_verts[i]:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    return outer_side + 50.0
        return outer_side
    
    bounds = [
        (-5.0, 5.0),    # outer_cx
        (-5.0, 5.0),    # outer_cy
        (4.0, 15.0),    # outer_side
        (0.0, 60.0)     # outer_angle
    ]
    
    best_side = float('inf')
    best_params = [0.0, 0.0, 10.0, 0.0]  # fallback
    
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=80,
            popsize=12,
            tol=1e-5,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=True
        )
        if result.fun < best_side:
            best_side = result.fun
            best_params = list(result.x)
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        # Use fallback if too close to deadline
        best_params = [0.0, 0.0, 10.0, 0.0]
    
    outer_cx, outer_cy, outer_side, outer_angle = best_params
    
    inner = [[inner_positions[i][0], inner_positions[i][1], 0.0] for i in range(N)]
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
