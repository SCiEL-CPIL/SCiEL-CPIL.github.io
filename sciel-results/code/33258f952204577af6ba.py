import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    sqrt3 = np.sqrt(3)
    
    # Initial positions: Center + 6 + 4
    pos = [[0.0, 0.0]]
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        pos.append([x, y])
        
    for i in range(4):
        angle = i * np.pi / 2
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        pos.append([x, y])
        
    pos = pos[:N]
    pos = np.array(pos)
    
    def check_collision(p1, p2):
        d = np.linalg.norm(p1 - p2)
        return d < sqrt3 - 1e-6
        
    def calc_L(pos):
        max_dist = 0
        for x, y in pos:
            d = np.sqrt(x*x + y*y)
            max_dist = max(max_dist, d + 1.0)
        return max_dist
    
    current_L = calc_L(pos)
    best_L = current_L
    best_pos = pos.copy()
    
    # Simple local search
    step_size = 0.1
    while time.time() < deadline:
        improved = False
        for i in range(N):
            # Try moving towards center
            direction = -pos[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            else:
                continue
            
            new_pos_i = pos[i] + direction * step_size
            new_pos = pos.copy()
            new_pos[i] = new_pos_i
            
            # Check collision with others
            valid = True
            for j in range(N):
                if i == j: continue
                if check_collision(new_pos[i], new_pos[j]):
                    valid = False
                    break
            
            if valid:
                new_L = calc_L(new_pos)
                if new_L < best_L - 1e-6:
                    best_L = new_L
                    best_pos = new_pos
                    pos = new_pos
                    improved = True
                    if time.time() > deadline - 10.0:
                        break
        if not improved:
            # Try smaller step or break
            step_size *= 0.5
            if step_size < 0.001:
                break
                
    # Format output
    inner = []
    for x, y in best_pos:
        inner.append([float(x), float(y), 0.0])
        
    return inner, (0.0, 0.0), float(best_L), 0.0
# EVOLVE-BLOCK-END
