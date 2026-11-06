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
    
    current_colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            current_colors[idx] = i
            idx += 1
            
    counts = np.zeros((NUM_CYCLES, MAX_COLOR), dtype=np.int8)
    distinct_counts = np.zeros(NUM_CYCLES, dtype=np.int8)
    
    for c_idx, cycle in enumerate(CYCLES):
        colors_in_cycle = [current_colors[e] for e in cycle]
        distinct_counts[c_idx] = len(set(colors_in_cycle))
        for col in colors_in_cycle:
            counts[c_idx, col] += 1
            
    global_color_counts = [0] * MAX_COLOR
    for c in current_colors:
        global_color_counts[c] += 1
        
    current_score = sum(1 for c in global_color_counts if c > 0)
    best_colors = current_colors[:]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        new_color = random.randint(0, MAX_COLOR - 1)
        
        if new_color == old_color:
            continue
            
        current_colors[e] = new_color
        
        valid = True
        affected_cycles = edge_to_cycles[e]
        
        for c_idx in affected_cycles:
            counts[c_idx, old_color] -= 1
            if counts[c_idx, old_color] == 0:
                distinct_counts[c_idx] -= 1
                
            if counts[c_idx, new_color] == 0:
                distinct_counts[c_idx] += 1
            counts[c_idx, new_color] += 1
            
            if distinct_counts[c_idx] == 4:
                valid = False
        
        if valid:
            old_count = global_color_counts[old_color]
            new_count = global_color_counts[new_color]
            
            delta_score = 0
            if new_count == 0: delta_score += 1
            if old_count == 1: delta_score -= 1
            
            new_score = current_score + delta_score
            
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors[:]
                
            delta = new_score - current_score
            if delta >= 0:
                current_score = new_score
                global_color_counts[old_color] -= 1
                global_color_counts[new_color] += 1
            else:
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                    global_color_counts[old_color] -= 1
                    global_color_counts[new_color] += 1
                else:
                    current_colors[e] = old_color
                    for c_idx in affected_cycles:
                        counts[c_idx, new_color] -= 1
                        if counts[c_idx, new_color] == 0:
                            distinct_counts[c_idx] -= 1
                        counts[c_idx, old_color] += 1
                        if counts[c_idx, old_color] == 1:
                            distinct_counts[c_idx] += 1
        else:
            current_colors[e] = old_color
            for c_idx in affected_cycles:
                counts[c_idx, new_color] -= 1
                if counts[c_idx, new_color] == 0:
                    distinct_counts[c_idx] -= 1
                counts[c_idx, old_color] += 1
                if counts[c_idx, old_color] == 1:
                    distinct_counts[c_idx] += 1
                    
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_colors
