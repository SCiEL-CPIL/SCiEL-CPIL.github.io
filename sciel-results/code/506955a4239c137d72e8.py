import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    num_k4s = len(k4s)
    
    # For each edge, which K4s does it appear in?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)
    
    def compact(colors):
        unique = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in colors], len(unique)
    
    # Try direct construction with (i+j) mod p for various p
    best_colors = None
    best_count = float('inf')
    
    for p in [13, 14, 15, 16, 17, 18, 19, 20]:
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (i + j) % p
        
        valid = True
        for k4_idx in range(num_k4s):
            e_list = k4s[k4_idx]
            cset = set(colors[ei] for ei in e_list)
            if len(cset) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break
    
    if best_colors is not None:
        return compact(best_colors)[0]
    
    # Optimized simulated annealing with O(1) incremental penalty updates
    def init_penalty_state(colors):
        k4_color_counts = [{} for _ in range(num_k4s)]
        penalty = 0
        for k4_idx in range(num_k4s):
            e_list = k4s[k4_idx]
            counts = {}
            for ei in e_list:
                c = colors[ei]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_idx] = counts
            if len(counts) < 5:
                penalty += 1
        return k4_color_counts, penalty
    
    def get_delta_penalty(colors, e_idx, old_c, new_c, k4_color_counts):
        delta = 0
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            old_distinct = len(counts)
            
            counts[old_c] -= 1
            if counts[old_c] == 0:
                del counts[old_c]
            
            if new_c not in counts:
                counts[new_c] = 0
            counts[new_c] += 1
            
            new_distinct = len(counts)
            
            old_bad = old_distinct < 5
            new_bad = new_distinct < 5
            if old_bad and not new_bad:
                delta -= 1
            elif not old_bad and new_bad:
                delta += 1
        
        return delta
    
    def apply_change(colors, e_idx, old_c, new_c, k4_color_counts):
        colors[e_idx] = new_c
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            counts[old_c] -= 1
            if counts[old_c] == 0:
                del counts[old_c]
            if new_c not in counts:
                counts[new_c] = 0
            counts[new_c] += 1
    
    def revert_change(colors, e_idx, old_c, new_c, k4_color_counts):
        colors[e_idx] = old_c
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            counts[new_c] -= 1
            if counts[new_c] == 0:
                del counts[new_c]
            if old_c not in counts:
                counts[old_c] = 0
            counts[old_c] += 1
    
    # Best solution tracking
    best_penalty = float('inf')
    best_count = float('inf')
    best_colors = None
    
    # Initial solutions to try
    initial_solutions = []
    for p in [13, 14, 15, 16, 17]:
        init = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            init[idx] = (i + j) % p
        initial_solutions.append(init)
    
    for _ in range(3):
        init = [random.randint(0, 14) for _ in range(num_edges)]
        initial_solutions.append(init)
    
    # Simulated annealing parameters
    T_max = 5.0
    T_min = 0.05
    cooling_rate = 0.9995
    
    for init_sol in initial_solutions:
        if time.time() > deadline - 10.0:
            break
        
        colors = init_sol[:]
        k4_color_counts, current_penalty = init_penalty_state(colors)
        current_count = len(set(colors))
        
        if current_penalty < best_penalty or (current_penalty == best_penalty and current_count < best_count):
            best_penalty = current_penalty
            best_count = current_count
            best_colors = colors[:]
            if best_penalty == 0 and best_count <= 13:
                break
        
        T = T_max
        iterations = 0
        while T > T_min and time.time() < deadline - 10.0:
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            
            if random.random() < 0.6:
                new_c = random.randint(0, 14)
            else:
                new_c = (old_c + random.randint(-3, 3)) % 15
            
            if new_c == old_c:
                T *= cooling_rate
                continue
            
            delta = get_delta_penalty(colors, e_idx, old_c, new_c, k4_color_counts)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                apply_change(colors, e_idx, old_c, new_c, k4_color_counts)
                current_penalty += delta
                new_count = len(set(colors))
                
                if current_penalty < best_penalty or (current_penalty == best_penalty and new_count < best_count):
                    best_penalty = current_penalty
                    best_count = new_count
                    best_colors = colors[:]
                    if best_penalty == 0 and best_count <= 13:
                        break
            else:
                pass
            
            T *= cooling_rate
            iterations += 1
    
    # Final polish: try to reduce color count if penalty is 0
    if best_colors is not None and best_penalty == 0 and best_count > 13:
        colors = best_colors[:]
        k4_color_counts, current_penalty = init_penalty_state(colors)
        current_count = best_count
        
        for _ in range(50000):
            if time.time() > deadline - 3.0:
                break
            
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            
            used_colors = list(set(colors))
            if len(used_colors) <= 13:
                break
            candidates = [c for c in used_colors if c != old_c]
            if not candidates:
                break
            new_c = random.choice(candidates)
            
            delta = get_delta_penalty(colors, e_idx, old_c, new_c, k4_color_counts)
            if delta <= 0:
                apply_change(colors, e_idx, old_c, new_c, k4_color_counts)
                current_penalty += delta
                new_count = len(set(colors))
                if new_count < best_count:
                    best_count = new_count
                    best_colors = colors[:]
    
    return compact(best_colors)[0]
