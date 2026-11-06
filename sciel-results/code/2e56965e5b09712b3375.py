import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Define the pool of candidate points
    # A grid from -10 to 10 provides 21x21 = 441 points
    pool = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            pool.append([x, y])
            
    n_pool = len(pool)
    
    # Pre-compute line keys for all pairs in the pool
    line_keys = [[None for _ in range(n_pool)] for _ in range(n_pool)]
    
    def get_line_key(i, j):
        if i == j:
            return None
        if line_keys[i][j] is not None:
            return line_keys[i][j]
        
        x1, y1 = pool[i]
        x2, y2 = pool[j]
        
        a = y1 - y2
        b = x2 - x1
        c = -(a * x1 + b * y1)
        
        g = math.gcd(a, b)
        if g != 0:
            g = math.gcd(g, c)
        else:
            g = abs(c) if c != 0 else 1
            
        if g < 0:
            g = -g
            
        if g != 0:
            a //= g
            b //= g
            c //= g
            
        # Ensure unique representation
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
            
        key = (a, b, c)
        line_keys[i][j] = key
        line_keys[j][i] = key
        return key

    # Reusable counter
    line_counts = defaultdict(int)

    # Evaluate a configuration (list of 16 indices)
    def evaluate(config):
        line_counts.clear()
        for i in range(16):
            idx_i = config[i]
            for j in range(i + 1, 16):
                idx_j = config[j]
                key = get_line_key(idx_i, idx_j)
                if key is not None:
                    line_counts[key] += 1
                    
        # Check for 5+ collinear
        max_count = 0
        four_count = 0
        for count in line_counts.values():
            if count > max_count:
                max_count = count
            if count == 4:
                four_count += 1
                
        if max_count >= 5:
            return -1 # Invalid
        return four_count

    # Initialize with a 4x4 grid
    init_x = [-2, -1, 0, 1]
    init_y = [-2, -1, 0, 1]
    
    init_points = []
    for x in init_x:
        for y in init_y:
            init_points.append([x, y])
            
    point_to_idx = {tuple(p): i for i, p in enumerate(pool)}
    current_config = [point_to_idx[tuple(p)] for p in init_points]
    current_set = set(current_config)
    
    current_score = evaluate(current_config)
    best_config = current_config[:]
    best_score = current_score
    
    # Simulated Annealing
    random.seed(42)
    
    T = 1.0
    T_min = 0.01
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random point to change
        i = random.randint(0, 15)
        old_idx = current_config[i]
        
        # Pick a new random point from pool, not in current config
        new_idx = random.randint(0, n_pool - 1)
        if new_idx in current_set:
            continue
            
        # Apply move
        current_config[i] = new_idx
        current_set.remove(old_idx)
        current_set.add(new_idx)
        
        new_score = evaluate(current_config)
        
        # Acceptance
        accepted = False
        if new_score > current_score:
            accepted = True
        elif new_score >= 0:
            delta = new_score - current_score
            if delta < 0:
                # Avoid underflow
                if T > 0:
                    prob = math.exp(delta / T)
                else:
                    prob = 0
                if random.random() < prob:
                    accepted = True
            else:
                accepted = True # Should not happen if new_score <= current_score, but safe
                
        if accepted:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_config = current_config[:]
                no_improve_count = 0
        else:
            # Reject
            current_config[i] = old_idx
            current_set.remove(new_idx)
            current_set.add(old_idx)
            no_improve_count += 1
            
        # Cooling / Restart logic
        if no_improve_count > 5000:
            # No improvement for a while, cool down or restart
            T *= 0.8
            if T < T_min:
                # Restart from best
                current_config = best_config[:]
                current_set = set(current_config)
                current_score = best_score
                T = 1.0
                no_improve_count = 0
            else:
                no_improve_count = 0
        else:
            # Slow cooling
            T = max(T_min, T * 0.99999)

    # Return best found
    best_points = [pool[i] for i in best_config]
    return best_points
# EVOLVE-BLOCK-END
