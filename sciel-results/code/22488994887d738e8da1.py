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

# Precompute cycles
CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

NUM_CYCLES = len(CYCLES)
cycle_edges = np.array(CYCLES, dtype=np.int32)

# Precompute edge to cycle mapping
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for c_idx in range(NUM_CYCLES):
    for e in cycle_edges[c_idx]:
        edge_to_cycles[e].append(c_idx)

edge_to_cycle_indices = [np.array(lst, dtype=np.int32) for lst in edge_to_cycles]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline with 5s safety margin
    
    best_score = 0
    best_colors = None
    
    # Deterministic valid starting point: color(i,j) = i
    # This uses N-1 = 15 colors and is guaranteed valid
    def make_deterministic():
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                colors[idx] = i
                idx += 1
        return colors
    
    current_colors = make_deterministic()
    global_color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
    for c in current_colors:
        global_color_counts[c] += 1
    current_score = np.count_nonzero(global_color_counts)
    
    if current_score > best_score:
        best_score = current_score
        best_colors = current_colors.copy()
    
    # Simulated annealing with smart moves
    # Statistical physics model: Energy = -Score (maximize score)
    # Hard constraint: No rainbow cycles
    T = 1.0  # Start with moderate temperature
    T_min = 1e-4
    alpha = 0.9995
    
    no_improve_count = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Choose edge to modify
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Generate candidate colors, prioritizing unused or low-frequency colors
        # to maximize color count (entropy)
        unused_colors = [c for c in range(MAX_COLOR) if global_color_counts[c] == 0]
        low_freq_colors = [c for c in range(MAX_COLOR) if 0 < global_color_counts[c] <= 2]
        
        # High temperature phase: explore unused colors more aggressively
        if unused_colors:
            if random.random() < 0.8:
                new_color = random.choice(unused_colors)
            else:
                new_color = random.randint(0, MAX_COLOR - 1)
        elif low_freq_colors and random.random() < 0.5:
            new_color = random.choice(low_freq_colors)
        else:
            new_color = random.randint(0, MAX_COLOR - 1)
        
        if new_color == old_color:
            continue
        
        # Check validity using vectorized operations
        affected_indices = edge_to_cycle_indices[e]
        if len(affected_indices) > 0:
            edges_in_cycles = cycle_edges[affected_indices]
            
            # Save old color, temporarily update
            current_colors[e] = new_color
            
            colors_in_cycles = current_colors[edges_in_cycles]
            
            # Check for rainbow (4 distinct colors)
            sorted_colors = np.sort(colors_in_cycles, axis=1)
            is_rainbow = (sorted_colors[:, 0] != sorted_colors[:, 1]) & \
                         (sorted_colors[:, 1] != sorted_colors[:, 2]) & \
                         (sorted_colors[:, 2] != sorted_colors[:, 3])
            
            valid = not np.any(is_rainbow)
        else:
            valid = True
        
        if valid:
            old_count = global_color_counts[old_color]
            new_count = global_color_counts[new_color]
            
            delta_score = 0
            if new_count == 0:
                delta_score += 1
            if old_count == 1:
                delta_score -= 1
            
            new_score = current_score + delta_score
            
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Acceptance criterion: Metropolis-Hastings
            # Energy difference = -delta_score (we maximize score, so minimize -score)
            # Accept if delta_score > 0 or with probability exp(delta_score / T)
            if delta_score >= 0:
                current_score = new_score
                global_color_counts[old_color] -= 1
                global_color_counts[new_color] += 1
            else:
                # delta_score is negative, so exp(delta_score / T) < 1
                if random.random() < math.exp(delta_score / max(T, 1e-10)):
                    current_score = new_score
                    global_color_counts[old_color] -= 1
                    global_color_counts[new_color] += 1
                else:
                    # Revert
                    current_colors[e] = old_color
        else:
            current_colors[e] = old_color
            no_improve_count += 1
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Final local search pass focusing on unused colors
    if time.time() < deadline - 5.0:
        for c in range(MAX_COLOR):
            if global_color_counts[c] == 0 and time.time() >= deadline - 5.0:
                break
            
            # Find edges where we can safely change to color c
            for e in range(NUM_EDGES):
                if current_colors[e] == c:
                    continue
                
                affected_indices = edge_to_cycle_indices[e]
                if len(affected_indices) == 0:
                    continue
                
                edges_in_cycles = cycle_edges[affected_indices]
                current_colors[e] = c
                colors_in_cycles = current_colors[edges_in_cycles]
                
                sorted_colors = np.sort(colors_in_cycles, axis=1)
                is_rainbow = (sorted_colors[:, 0] != sorted_colors[:, 1]) & \
                             (sorted_colors[:, 1] != sorted_colors[:, 2]) & \
                             (sorted_colors[:, 2] != sorted_colors[:, 3])
                
                if not np.any(is_rainbow):
                    # Valid change
                    old_color = current_colors[e]
                    old_count = global_color_counts[old_color]
                    global_color_counts[old_color] -= 1
                    global_color_counts[c] += 1
                    current_score += 1
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = current_colors.copy()
                else:
                    current_colors[e] = current_colors[e]  # no change
                    
                if time.time() >= deadline - 5.0:
                    break
    
    return best_colors.tolist() if best_colors is not None else [0] * NUM_EDGES
