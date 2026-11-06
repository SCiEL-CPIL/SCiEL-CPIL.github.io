import time
import random
import math
from itertools import combinations

def construct():
    deadline = time.time() + 115  # Safety margin
    
    n = 16
    edge_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_idx)
    
    # Map edge (i,j) to index
    edge_to_idx = {}
    for idx, (i, j) in enumerate(edge_idx):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
        
    # Precompute all K4s and their edge indices
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in combinations(k4, 2):
            edges.append(edge_to_idx[(a, b)])
        k4_edges.append(edges)
        
    # For each edge, which K4s does it belong to?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)

    def setup_state(colors):
        k4_color_counts = [{} for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        is_violated = [False] * num_k4s
        violation_count = 0
        global_color_counts = [0] * 120
        
        for c in colors:
            if c < len(global_color_counts):
                global_color_counts[c] += 1
        
        for k4_idx, edges in enumerate(k4_edges):
            counts = {}
            for e in edges:
                c = colors[e]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_idx] = counts
            k4_distinct[k4_idx] = len(counts)
            if k4_distinct[k4_idx] < 5:
                is_violated[k4_idx] = True
                violation_count += 1
                
        return k4_color_counts, k4_distinct, is_violated, violation_count, global_color_counts

    def get_unique_color_count(global_color_counts, max_color):
        return sum(1 for c in range(max_color) if global_color_counts[c] > 0)

    def simulated_annealing(init_colors, num_c, time_limit):
        colors = init_colors[:]
        k4_color_counts, k4_distinct, is_violated, violation_count, global_color_counts = setup_state(colors)
        
        current_violations = violation_count
        current_unique = get_unique_color_count(global_color_counts, num_c)
        
        best_violations = current_violations
        best_unique = current_unique
        best_colors = colors[:]
        
        T = 1.0
        cooling = 0.9995
        
        while time.time() < time_limit:
            if current_violations == 0:
                # Try to reduce unique colors
                e = random.randint(0, num_edges - 1)
                c_old = colors[e]
                c_new = random.randint(0, num_c - 1)
                
                if c_new == c_old:
                    continue
                    
                # Calculate delta
                affected_k4s = edge_k4s[e]
                delta_violations = 0
                delta_unique = 0
                
                if global_color_counts[c_old] == 1:
                    delta_unique -= 1
                if global_color_counts[c_new] == 0:
                    delta_unique += 1
                
                can_move = True
                for k_idx in affected_k4s:
                    k_counts = k4_color_counts[k_idx]
                    k_dist = k4_distinct[k_idx]
                    
                    cnt_old = k_counts.get(c_old, 0)
                    cnt_new = k_counts.get(c_new, 0)
                    
                    new_dist = k_dist
                    if cnt_old == 1:
                        new_dist -= 1
                    if cnt_new == 0:
                        new_dist += 1
                        
                    if new_dist < 5:
                        if k_dist >= 5:
                            delta_violations += 1
                        elif new_dist >= 5:
                            delta_violations -= 1
                            
                if delta_violations == 0 and delta_unique <= 0:
                    # Accept
                    colors[e] = c_new
                    global_color_counts[c_old] -= 1
                    global_color_counts[c_new] += 1
                    
                    for k_idx in affected_k4s:
                        k_counts = k4_color_counts[k_idx]
                        k_dist = k4_distinct[k_idx]
                        
                        cnt_old = k_counts.get(c_old, 0)
                        cnt_new = k_counts.get(c_new, 0)
                        
                        new_dist = k_dist
                        if cnt_old == 1:
                            del k_counts[c_old]
                            new_dist -= 1
                        else:
                            k_counts[c_old] = cnt_old - 1
                            
                        if cnt_new == 0:
                            k_counts[c_new] = 1
                            new_dist += 1
                        else:
                            k_counts[c_new] = cnt_new + 1
                            
                        k4_distinct[k_idx] = new_dist
                        
                    current_unique += delta_unique
                    if current_unique < best_unique:
                        best_unique = current_unique
                        best_colors = colors[:]
                        
            else:
                # Not valid, try to reduce violations
                # Pick a random K4 until we find a violated one
                v_idx = -1
                for _ in range(10):
                    v_idx = random.randint(0, num_k4s - 1)
                    if is_violated[v_idx]:
                        break
                if v_idx == -1:
                    # Should not happen if violation_count > 0
                    continue
                    
                v_k4_edges = k4_edges[v_idx]
                e = random.choice(v_k4_edges)
                c_old = colors[e]
                c_new = random.randint(0, num_c - 1)
                
                if c_new == c_old:
                    continue
                    
                # Simulate
                affected_k4s = edge_k4s[e]
                delta_violations = 0
                
                for k_idx in affected_k4s:
                    k_counts = k4_color_counts[k_idx]
                    k_dist = k4_distinct[k_idx]
                    was_violated = is_violated[k_idx]
                    
                    cnt_old = k_counts.get(c_old, 0)
                    cnt_new = k_counts.get(c_new, 0)
                    
                    new_dist = k_dist
                    if cnt_old == 1:
                        new_dist -= 1
                    if cnt_new == 0:
                        new_dist += 1
                        
                    is_violated_new = new_dist < 5
                    
                    if was_violated and not is_violated_new:
                        delta_violations -= 1
                    elif not was_violated and is_violated_new:
                        delta_violations += 1
                        
                # Acceptance criterion
                if delta_violations < 0:
                    accept = True
                elif delta_violations == 0:
                    delta_unique = 0
                    if global_color_counts[c_old] == 1:
                        delta_unique -= 1
                    if global_color_counts[c_new] == 0:
                        delta_unique += 1
                    accept = (delta_unique < 0) or (random.random() < 0.5)
                else:
                    prob = math.exp(-delta_violations / T)
                    accept = (random.random() < prob)
                    
                if accept:
                    # Apply
                    colors[e] = c_new
                    global_color_counts[c_old] -= 1
                    global_color_counts[c_new] += 1
                    
                    for k_idx in affected_k4s:
                        k_counts = k4_color_counts[k_idx]
                        k_dist = k4_distinct[k_idx]
                        was_violated = is_violated[k_idx]
                        
                        cnt_old = k_counts.get(c_old, 0)
                        cnt_new = k_counts.get(c_new, 0)
                        
                        new_dist = k_dist
                        if cnt_old == 1:
                            del k_counts[c_old]
                            new_dist -= 1
                        else:
                            k_counts[c_old] = cnt_old - 1
                            
                        if cnt_new == 0:
                            k_counts[c_new] = 1
                            new_dist += 1
                        else:
                            k_counts[c_new] = cnt_new + 1
                            
                        is_violated_new = new_dist < 5
                        k4_distinct[k_idx] = new_dist
                        
                        if was_violated and not is_violated_new:
                            is_violated[k_idx] = False
                            current_violations -= 1
                        elif not was_violated and is_violated_new:
                            is_violated[k_idx] = True
                            current_violations += 1
                            
                    current_unique = get_unique_color_count(global_color_counts, num_c)
                    
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_unique = current_unique
                        best_colors = colors[:]
                    elif current_violations == 0 and current_unique < best_unique:
                        best_unique = current_unique
                        best_colors = colors[:]
                        
            T *= cooling
            if T < 0.01:
                break
                
        return best_colors, best_violations, best_unique

    # Generate 16-color multiplicative solution
    v = list(range(1, 17))
    base_colors = [(v[i] * v[j]) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    best_result = base_colors
    best_colors_used = len(set(base_colors))
    
    # Try to reduce colors
    for num_c in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        # Strategy 1: Start from base_colors, map to num_c
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            mapping = [random.randint(0, num_c - 1) for _ in range(16)]
            init_colors = [mapping[c] for c in base_colors]
            
            res_colors, res_viol, res_unique = simulated_annealing(init_colors, num_c, deadline - 5)
            
            if res_viol == 0:
                if res_unique < best_colors_used:
                    best_colors_used = res_unique
                    best_result = res_colors
                    
        # Strategy 2: Random initialization
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            init_colors = [random.randint(0, num_c - 1) for _ in range(num_edges)]
            res_colors, res_viol, res_unique = simulated_annealing(init_colors, num_c, deadline - 5)
            
            if res_viol == 0:
                if res_unique < best_colors_used:
                    best_colors_used = res_unique
                    best_result = res_colors

    if best_colors_used > len(set(base_colors)):
        return base_colors
        
    return best_result
