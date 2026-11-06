import time
import random
import math
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

NUM_CYCLES = len(CYCLES)

edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for c_idx, cycle in enumerate(CYCLES):
    for e in cycle:
        edge_to_cycles[e].append(c_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    def greedy_init():
        colors = [-1] * NUM_EDGES
        # Shuffle edges
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        # Precompute valid colors? No, dynamic.
        # For speed, we check validity on the fly.
        
        used_colors = set()
        
        for e in edges:
            # Try unused colors first
            found = False
            # We want to try low indices of unused colors?
            # Or random? Random might explore more.
            # Let's try random unused colors.
            unused = [c for c in range(MAX_COLOR) if c not in used_colors]
            random.shuffle(unused)
            
            for c in unused:
                if is_valid_move(e, c, colors):
                    colors[e] = c
                    used_colors.add(c)
                    found = True
                    break
            
            if not found:
                # Try used colors
                # Pick a random used color
                if used_colors:
                    used_list = list(used_colors)
                    random.shuffle(used_list)
                    for c in used_list:
                        if is_valid_move(e, c, colors):
                            colors[e] = c
                            found = True
                            break
                if not found:
                    # Fallback: color 0
                    colors[e] = 0
                    used_colors.add(0)
                    
        return colors

    def is_valid_move(e, c, colors):
        # Check if setting colors[e] = c creates a 4-distinct cycle
        # We need to check all cycles containing e.
        # But colors[e] is currently -1 (or old value in SA).
        # In greedy, it's -1.
        # We simulate the change.
        # To avoid modifying `colors`, we pass `colors` and check.
        # But `colors` has -1 for unset edges.
        # The cycle check needs the colors of the other 3 edges.
        # If any other edge in the cycle is -1, the cycle is not yet "complete" with 4 distinct colors?
        # Actually, if an edge is -1, it doesn't contribute a color yet.
        # So a cycle is only invalid if it has 4 DISTINCT non-negative colors.
        # If we set e to c, and the other 3 edges are, say, 1, 2, 3, and c=4, then we have 4 distinct.
        # If the other 3 are 1, 1, 1, and c=4, we have 2 distinct.
        
        for c_idx in edge_to_cycles[e]:
            cycle = CYCLES[c_idx]
            cols = set()
            for e_i in cycle:
                if e_i == e:
                    cols.add(c)
                else:
                    col = colors[e_i]
                    if col != -1:
                        cols.add(col)
            if len(cols) == 4:
                return False
        return True

    def run_sa(init_colors, time_limit):
        colors = init_colors[:]
        counts = np.zeros((NUM_CYCLES, MAX_COLOR), dtype=np.int8)
        distinct_counts = np.zeros(NUM_CYCLES, dtype=np.int8)
        
        global_color_counts = [0] * MAX_COLOR
        for c in colors:
            global_color_counts[c] += 1
            
        current_score = sum(1 for c in global_color_counts if c > 0)
        
        for c_idx, cycle in enumerate(CYCLES):
            cs = [colors[e] for e in cycle]
            distinct_counts[c_idx] = len(set(cs))
            for col in cs:
                counts[c_idx, col] += 1
                
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            
            # Strategy: Bias towards new colors?
            # If we pick a random color, we might miss new ones.
            # Let's pick a random color from 0..MAX_COLOR-1.
            new_color = random.randint(0, MAX_COLOR - 1)
            
            if new_color == old_color:
                continue
            
            # Check validity
            valid = True
            for c_idx in edge_to_cycles[e]:
                # Update counts temporarily?
                # Or just check distinct counts logic.
                # counts[c_idx, old_color] will decrease.
                # counts[c_idx, new_color] will increase.
                
                old_cnt = counts[c_idx, old_color]
                new_cnt = counts[c_idx, new_color]
                
                # Calculate new distinct count
                # If old_cnt == 1, old_color disappears -> distinct - 1
                # If new_cnt == 0, new_color appears -> distinct + 1
                # Else distinct unchanged.
                
                delta_distinct = 0
                if old_cnt == 1:
                    delta_distinct -= 1
                if new_cnt == 0:
                    delta_distinct += 1
                    
                new_distinct = distinct_counts[c_idx] + delta_distinct
                if new_distinct == 4:
                    valid = False
                    break
            
            if valid:
                # Accept if improves or with probability
                old_global = global_color_counts[old_color]
                new_global = global_color_counts[new_color]
                
                delta_score = 0
                if new_global == 0: delta_score += 1
                if old_global == 1: delta_score -= 1
                
                new_score = current_score + delta_score
                
                if delta_score >= 0:
                    # Always accept improvements or neutral
                    current_score = new_score
                    global_color_counts[old_color] -= 1
                    global_color_counts[new_color] += 1
                    
                    for c_idx in edge_to_cycles[e]:
                        counts[c_idx, old_color] -= 1
                        if counts[c_idx, old_color] == 0:
                            distinct_counts[c_idx] -= 1
                        counts[c_idx, new_color] += 1
                        if counts[c_idx, new_color] == 1:
                            distinct_counts[c_idx] += 1
                            
                    colors[e] = new_color
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        
                else:
                    # Accept with prob exp(delta/T)
                    if random.random() < math.exp(delta_score / T):
                        current_score = new_score
                        global_color_counts[old_color] -= 1
                        global_color_counts[new_color] += 1
                        
                        for c_idx in edge_to_cycles[e]:
                            counts[c_idx, old_color] -= 1
                            if counts[c_idx, old_color] == 0:
                                distinct_counts[c_idx] -= 1
                            counts[c_idx, new_color] += 1
                            if counts[c_idx, new_color] == 1:
                                distinct_counts[c_idx] += 1
                            
                        colors[e] = new_color
                    # Else reject (do nothing)
            # If invalid, reject (do nothing)
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return colors, current_score

    # Multi-start
    while time.time() < deadline:
        init = greedy_init()
        score = sum(1 for c in set(init) if c != -1)
        if score > best_score:
            best_score = score
            best_colors = init[:]
            
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Run SA for a short duration
        _, score = run_sa(init, 1.0)
        
        if time.time() > deadline:
            break

    # Final check
    if best_colors[0] == -1:
         return [0] * NUM_EDGES
         
    return best_colors
