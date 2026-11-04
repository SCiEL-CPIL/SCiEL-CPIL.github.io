import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3)
HEX_APOthem = SQRT3 / 2.0

UNIT_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, SQRT3/2],
    [-0.5, SQRT3/2],
    [-1.0, 0.0],
    [-0.5, -SQRT3/2],
    [0.5, -SQRT3/2]
])

OUTER_NORMALS = np.array([
    [math.cos(math.pi/6), math.sin(math.pi/6)],
    [0.0, 1.0],
    [-math.cos(math.pi/6), math.sin(math.pi/6)],
    [-math.cos(math.pi/6), -math.sin(math.pi/6)],
    [0.0, -1.0],
    [math.cos(math.pi/6), -math.sin(math.pi/6)]
])

def initial_positions():
    centers = []
    # Row 0: 3 hexes
    for i in range(3):
        centers.append([2.0 * (i - 1), 0.0])
    # Row 1: 4 hexes
    for i in range(4):
        centers.append([2.0 * (i - 1.5), SQRT3])
    # Row 2: 3 hexes
    for i in range(3):
        centers.append([2.0 * (i - 1), 2.0 * SQRT3])
    # Row 3: 2 hexes
    for i in range(2):
        centers.append([2.0 * (i - 0.5), 3.0 * SQRT3])
    return np.array(centers)

def check_containment(centers, S):
    max_violation = 0.0
    limit = S * SQRT3 / 2.0
    for i in range(N):
        c = centers[i]
        verts = UNIT_HEX_VERTS + c
        dots = np.dot(verts, OUTER_NORMALS.T)
        max_dots = np.max(dots, axis=0)
        violations = np.max(max_dots - limit)
        if violations > max_violation:
            max_violation = violations
    return max_violation

def check_collisions(centers):
    penalty = 0.0
    min_dist = 2.0
    for i in range(N):
        for j in range(i+1, N):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += overlap * overlap
    return penalty

def calculate_penalty(centers, S):
    c_viol = check_containment(centers, S)
    c_coll = check_collisions(centers)
    return c_viol * 10.0 + c_coll

def simulated_annealing(S, time_limit, initial_centers):
    centers = initial_centers.copy()
    best_centers = centers.copy()
    best_penalty = calculate_penalty(centers, S)
    
    T = 1.0
    T_min = 1e-6
    step_size = 0.1
    
    start_time = time.time()
    current_penalty = best_penalty
    current_centers = centers
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(N)
        old_x, old_y = current_centers[i]
        new_x = old_x + np.random.normal(0, step_size)
        new_y = old_y + np.random.normal(0, step_size)
        
        if abs(new_x) > S + 1 or abs(new_y) > S + 1:
            continue
            
        current_centers[i] = [new_x, new_y]
        new_penalty = calculate_penalty(current_centers, S)
        
        delta = new_penalty - current_penalty
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_penalty = new_penalty
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_centers = current_centers.copy()
        else:
            current_centers[i] = [old_x, old_y]
            
        T *= 0.999
        if T < T_min:
            T = T_min
            
    return best_centers, best_penalty

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Baseline
    init_centers = initial_positions()
    best_S = 4.5
    best_config = init_centers
    
    # Verify baseline
    c_viol = check_containment(best_config, best_S)
    c_coll = check_collisions(best_config)
    if c_viol > 1e-6 or c_coll > 1e-6:
        # Should not happen, but just in case
        best_S = 5.0
        
    S_low = 3.4641
    S_high = best_S
    
    current_centers = best_config.copy()
    
    # Binary search
    while time.time() - start_time < 140.0:
        S_mid = (S_low + S_high) / 2.0
        
        # Time allocation for SA
        # We have ~140s total.
        # Let's run SA for 5 seconds per iteration.
        centers, penalty = simulated_annealing(S_mid, 5.0, current_centers)
        
        c_viol = check_containment(centers, S_mid)
        c_coll = check_collisions(centers)
        
        if c_viol < 1e-6 and c_coll < 1e-6:
            best_S = S_mid
            best_config = centers
            S_high = S_mid
        else:
            S_low = S_mid
            
        current_centers = centers
        
        # If we are very close to the lower bound, stop
        if S_high - S_low < 0.001:
            break
            
    inner = []
    for c in best_config:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    outer_center = [0.0, 0.0]
    outer_side = float(best_S)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
