import time
import random
import sys

N = 16
NUM_EDGES = 120

# Precompute K4 distances
# Distance d in 1..8
K4_DISTANCES = []

def get_distance(i, j):
    d = abs(i - j)
    if d > 8:
        d = 16 - d
    return d

# Generate all K4s
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                dists = [
                    get_distance(a, b),
                    get_distance(a, c),
                    get_distance(a, d),
                    get_distance(b, c),
                    get_distance(b, d),
                    get_distance(c, d)
                ]
                K4_DISTANCES.append(tuple(dists))

def evaluate(color_map):
    penalty = 0
    for dists in K4_DISTANCES:
        # Unroll or use set
        # Using set is fine
        c0 = color_map[dists[0]]
        c1 = color_map[dists[1]]
        c2 = color_map[dists[2]]
        c3 = color_map[dists[3]]
        c4 = color_map[dists[4]]
        c5 = color_map[dists[5]]
        
        # Check if at least 5 distinct
        # If all 6 are distinct, OK.
        # If 5 distinct, OK.
        # If 4 or less, penalty.
        
        # Simple way:
        if c0 == c1 == c2 == c3 == c4 == c5:
            penalty += 1
        elif c0 == c1 and c0 == c2 and c0 == c3 and c0 == c4 and c0 == c5:
             # This is covered by above? No.
             # Let's just use set for clarity and correctness, it's fast enough.
             pass
             
        # Actually, let's stick to set for safety.
        # But we can optimize by checking if the number of unique elements is < 5.
        # len(set(...)) < 5
        
        # Optimization:
        # If any two are different, we have at least 2.
        # We need < 5, so 1, 2, 3, 4.
        
        # Let's just use the set method, it's robust.
        if len({c0, c1, c2, c3, c4, c5}) < 5:
            penalty += 1
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    for K in [13, 14, 15]:
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            color_map = {d: random.randint(0, K - 1) for d in range(1, 9)}
            current_penalty = evaluate(color_map)
            
            if current_penalty == 0:
                colors = [0] * NUM_EDGES
                idx = 0
                for i in range(N):
                    for j in range(i + 1, N):
                        d = get_distance(i, j)
                        colors[idx] = color_map[d]
                        idx += 1
                return colors
            
            steps = 0
            max_steps = 2000
            while steps < max_steps and time.time() < deadline - 10:
                steps += 1
                d_to_change = random.randint(1, 8)
                old_color = color_map[d_to_change]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                
                color_map[d_to_change] = new_color
                new_penalty = evaluate(color_map)
                
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                    if current_penalty == 0:
                        colors = [0] * NUM_EDGES
                        idx = 0
                        for i in range(N):
                            for j in range(i + 1, N):
                                dd = get_distance(i, j)
                                colors[idx] = color_map[dd]
                                idx += 1
                        return colors
                else:
                    color_map[d_to_change] = old_color
    
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
