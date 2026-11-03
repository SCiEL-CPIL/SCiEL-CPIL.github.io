import numpy as np
import random
import math

N = 12

def hex_support_min_max(c, theta, n_vec):
    # Returns (max_proj, min_proj)
    nx, ny = n_vec
    proj_c = c[0] * nx + c[1] * ny
    
    angle = math.atan2(ny, nx) - theta
    phi_mod = angle % (math.pi / 3)
    delta = min(phi_mod, math.pi / 3 - phi_mod)
    
    apothem = math.sqrt(3) / 2
    local_h = apothem / math.cos(delta)
    
    max_val = proj_c + local_h
    min_val = proj_c - local_h
    return max_val, min_val

def calculate_side_length_fixed_outer(positions):
    # Outer hexagon angle 0
    # Normals at 0, 60, 120, 180, 240, 300
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    sqrt3 = math.sqrt(3)
    normals = [
        (1.0, 0.0),
        (0.5, sqrt3/2),
        (-0.5, sqrt3/2),
        (-1.0, 0.0),
        (-0.5, -sqrt3/2),
        (0.5, -sqrt3/2)
    ]
    
    A_max = 0
    for n_vec in normals:
        max_proj = -float('inf')
        min_proj = float('inf')
        
        for (x, y, inner_theta) in positions:
            c = (x, y)
            mx, mn = hex_support_min_max(c, inner_theta, n_vec)
            if mx > max_proj: max_proj = mx
            if mn < min_proj: min_proj = mn
            
        required_A = max(max_proj, -min_proj)
        if required_A > A_max:
            A_max = required_A
            
    return (2 / sqrt3) * A_max

def construct():
    # Init
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    init_points = []
    init_points.append(axial_to_cartesian(0, 0))
    ring1 = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    for q, r in ring1:
        init_points.append(axial_to_cartesian(q, r))
    ring2 = [(2,0), (-2,0), (0,2), (0,-2), (2,-2)]
    for q, r in ring2:
        init_points.append(axial_to_cartesian(q, r))
        
    positions = [[x, y, 0.0] for x, y in init_points]
    
    T = 1.0
    cooling = 0.99
    iterations = 3000
    step_size = 0.5
    
    current_pos = [list(p) for p in positions]
    current_score = calculate_side_length_fixed_outer(current_pos)
    best_pos = [list(p) for p in current_pos]
    best_score = current_score
    
    for i in range(iterations):
        idx = random.randint(0, N-1)
        dim = random.randint(0, 2)
        
        old_val = current_pos[idx][dim]
        if dim < 2:
            new_val = old_val + random.gauss(0, step_size)
        else:
            new_val = old_val + random.gauss(0, step_size / 10.0)
            
        current_pos[idx][dim] = new_val
        
        new_score = calculate_side_length_fixed_outer(current_pos)
        
        if new_score < current_score or random.random() < math.exp((current_score - new_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_pos = [list(p) for p in current_pos]
        else:
            current_pos[idx][dim] = old_val
            
        T *= cooling
        
    inner = [[p[0], p[1], math.degrees(p[2])] for p in best_pos]
    return inner, (0.0, 0.0), best_score, 0.0
