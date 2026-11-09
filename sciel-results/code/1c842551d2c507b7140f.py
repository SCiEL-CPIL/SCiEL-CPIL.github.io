import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (limit is 150s)

    # Precompute K4s
    def get_idx(i, j):
        return i * 15 - i * (i - 1) // 2 + (j - i - 1)
        
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
            
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    k4s_all = []
    edge_k4s = [[] for _ in range(120)]
    
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4_idx = len(k4s_all)
        k4s_all.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    def count_invalid_k4s(colors):
        invalid_count = 0
        for e_indices in k4s_all:
            c0 = colors[e_indices[0]]
            c1 = colors[e_indices[1]]
            c2 = colors[e_indices[2]]
            c3 = colors[e_indices[3]]
            c4 = colors[e_indices[4]]
            c5 = colors[e_indices[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                invalid_count += 1
        return invalid_count

    def try_find_valid(K, time_limit):
        # Simulated Annealing with smarter initialization and moves
        colors = [random.randint(0, K - 1) for _ in range(120)]
        
        # Calculate initial invalid count
        total_invalid = count_invalid_k4s(colors)
        
        best_colors = colors[:]
        best_invalid = total_invalid
        
        if best_invalid == 0:
            return best_colors
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < time_limit:
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            new_color = random.randint(0, K - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate delta in invalid count
            delta_invalid = 0
            
            # For each K4 containing e_idx
            for k4_idx in edge_k4s[e_idx]:
                # Check validity before
                e_indices = k4s_all[k4_idx]
                c0 = colors[e_indices[0]]
                c1 = colors[e_indices[1]]
                c2 = colors[e_indices[2]]
                c3 = colors[e_indices[3]]
                c4 = colors[e_indices[4]]
                c5 = colors[e_indices[5]]
                valid_before = len({c0, c1, c2, c3, c4, c5}) >= 5
                
                # Temporarily change
                colors[e_idx] = new_color
                
                c0 = colors[e_indices[0]]
                c1 = colors[e_indices[1]]
                c2 = colors[e_indices[2]]
                c3 = colors[e_indices[3]]
                c4 = colors[e_indices[4]]
                c5 = colors[e_indices[5]]
                valid_after = len({c0, c1, c2, c3, c4, c5}) >= 5
                
                if valid_before and not valid_after:
                    delta_invalid += 1
                elif not valid_before and valid_after:
                    delta_invalid -= 1
            
            new_total_invalid = total_invalid + delta_invalid
            
            # Acceptance criteria
            accept = False
            if new_total_invalid < total_invalid:
                accept = True
            elif new_total_invalid == total_invalid:
                accept = random.random() < 0.5
            else:
                delta = new_total_invalid - total_invalid
                if random.random() < pow(2.71828, -delta / T):
                    accept = True
            
            if accept:
                total_invalid = new_total_invalid
                
                if total_invalid < best_invalid:
                    best_invalid = total_invalid
                    best_colors = colors[:]
                    
                    if best_invalid == 0:
                        return best_colors
            else:
                colors[e_idx] = old_color
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_colors

    # Try K=15 first
    t_start = time.time()
    colors_15 = try_find_valid(15, t_start + 40)
    
    if count_invalid_k4s(colors_15) == 0:
        unique_colors = sorted(set(colors_15))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors_15]
        
    # Try K=16
    t_start = time.time()
    colors_16 = try_find_valid(16, t_start + 30)
    
    if count_invalid_k4s(colors_16) == 0:
        unique_colors = sorted(set(colors_16))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors_16]
        
    # Fallback to K=17
    colors_17 = [0] * 120
    for idx, (i, j) in enumerate(edges):
        colors_17[idx] = (i + j) % 17
        
    if count_invalid_k4s(colors_17) == 0:
        unique_colors = sorted(set(colors_17))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors_17]
        
    return colors_17
