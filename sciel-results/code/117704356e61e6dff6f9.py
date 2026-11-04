import math
import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # 1. Define the 12 unit hexagons in a honeycomb lattice
    # Unit hexagon vertices (flat-top, centered at origin)
    v = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Centers of 12 hexagons in a hexagonal cluster (radius 1)
    centers = []
    centers.append((0, 0))
    for k in range(6):
        angle = k * math.pi / 3
        centers.append((math.cos(angle), math.sqrt(3) * math.sin(angle)))
    
    # Generate vertices for all 12 hexagons
    all_vertices = []
    for cx, cy in centers:
        all_vertices.append(v + np.array([cx, cy]))
    all_vertices = np.vstack(all_vertices)
    
    # 2. Define the function to calculate the minimum enclosing regular hexagon side length
    def calc_side_length(theta, scale=1.0):
        # Rotate vertices
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        verts = (R @ all_vertices.T).T * scale
        
        # The minimum enclosing regular hexagon of a set of points (aligned with axes)
        # has side length L such that the vertices lie within the hexagon.
        # For a regular hexagon with side L, the inequality for a point (x, y) is:
        # |x| <= L, |y| <= L*sqrt(3)/2, |x + y/sqrt(3)| <= L, etc.
        # Specifically, the width of the hexagon in the 6 normal directions is 2*L*sqrt(3)/2 = L*sqrt(3).
        # The distance from center to side is L*sqrt(3)/2.
        # So we check the projection of vertices onto the 6 normal directions.
        
        # Normal directions for a regular hexagon (flat-top)
        # Normals are at angles 0, 60, 120, 180, 240, 300 degrees
        # n_i = (cos(i*pi/3), sin(i*pi/3))
        # The constraint is n_i . p <= L * sqrt(3) / 2 for all i and all p.
        # So L >= (2 / sqrt(3)) * max_i max_p (n_i . p)
        
        # Compute projections
        # n_i . p = x*cos(i*pi/3) + y*sin(i*pi/3)
        # We can compute this for all 6 directions efficiently
        # Let's just loop over the 6 directions, it's fast enough for 72 points
        max_proj = 0.0
        for i in range(6):
            angle = i * math.pi / 3
            nx = math.cos(angle)
            ny = math.sin(angle)
            projs = verts[:, 0] * nx + verts[:, 1] * ny
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
        
        return (2.0 / math.sqrt(3)) * max_proj

    # 3. Optimize rotation theta
    # The function f(theta) is periodic with period 30 degrees (pi/6) due to hexagonal symmetry.
    # We search in [0, pi/6].
    
    # First, do a coarse grid search to find the minimum region
    best_theta = 0.0
    best_val = float('inf')
    
    # Grid search
    num_grid = 100
    for i in range(num_grid):
        theta = i * (math.pi / 6) / num_grid
        val = calc_side_length(theta)
        if val < best_val:
            best_val = val
            best_theta = theta
        
        if time.time() > deadline - 10:
            break

    # Refine with Golden Section Search
    # We need to find the interval containing the minimum.
    # Since the function is smooth and unimodal in the relevant sector (likely),
    # we can use the grid result to set bounds or just run GS on the whole [0, pi/6].
    # However, GS requires unimodality. The function might have multiple local minima.
    # Given the symmetry, the global minimum is likely at a symmetric position or near it.
    # Let's try GS on the whole range [0, pi/6].
    
    gr = (math.sqrt(5) + 1) / 2
    inv_gr = gr - 1
    
    a = 0.0
    b = math.pi / 6
    
    c = b - inv_gr * (b - a)
    d = a + inv_gr * (b - a)
    
    fc = calc_side_length(c)
    fd = calc_side_length(d)
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        if abs(b - a) < 1e-10:
            break
            
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - inv_gr * (b - a)
            fc = calc_side_length(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + inv_gr * (b - a)
            fd = calc_side_length(d)
            
    best_theta = (a + b) / 2
    best_val = calc_side_length(best_theta)
    
    # 4. Construct the inner hexagons
    R = np.array([[math.cos(best_theta), -math.sin(best_theta)],
                  [math.sin(best_theta), math.cos(best_theta)]])
    
    inner = []
    for cx, cy in centers:
        # Rotate center
        new_cx, new_cy = R @ np.array([cx, cy])
        inner.append([new_cx, new_cy, math.degrees(best_theta)])
        
    # Outer hexagon parameters
    center = (0.0, 0.0)
    side = best_val
    angle = math.degrees(best_theta)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
