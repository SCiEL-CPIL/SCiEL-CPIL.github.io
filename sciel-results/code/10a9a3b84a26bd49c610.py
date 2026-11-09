import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Edge index mapping
    def get_idx(i, j):
        return i * 15 - i * (i - 1) // 2 + (j - i - 1)
        
    edges = []
    edge_idx_map = {}
    for i in range(16):
        for j in range(i + 1, 16):
            idx = get_idx(i, j)
            edges.append((i, j))
            edge_idx_map[(i, j)] = idx
            edge_idx_map[(j, i)] = idx
            
    # K4s
    k4s = []
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
        k4_idx = len(k4s)
        k4s.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    def count_distinct_colors(e_indices, colors):
        c_set = set()
        for ei in e_indices:
            c_set.add(colors[ei])
        return len(c_set)

    def calc_penalty(colors):
        penalty = 0
        for e_indices in k4s:
            dc = count_distinct_colors(e_indices, colors)
            if dc < 5:
                penalty += (5 - dc)
        return penalty

    # Base solution: 17 colors
    base_colors = [0] * 120
    for idx, (i, j) in enumerate(edges):
        base_colors[idx] = (i + j) % 17
        
    # Verify base solution is valid
    if calc_penalty(base_colors) != 0:
        # This shouldn't happen, but just in case
        pass

    # Try to find solutions with fewer colors
    # We will try K = 13, 14, 15
    # We use Simulated Annealing to minimize penalty
    # If penalty becomes 0, we have a valid solution with K colors
    
    for K in [13, 14, 15]:
        # Time budget for this K - prioritize K=13
        t_start = time.time()
        t_budget = 40.0 if K == 13 else (25.0 if K == 14 else 15.0)
        
        # Initialize colors with a structured approach to get closer to valid
        colors = [0] * 120
        for idx, (i, j) in enumerate(edges):
            # Use a hash-like function to distribute colors
            colors[idx] = (i * 7 + j * 3 + idx) % K
            
        current_penalty = calc_penalty(colors)
        
        if current_penalty == 0:
            return colors
            
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99985
        
        # Track best solution
        best_colors = colors[:]
        best_penalty = current_penalty
        
        # Local search: try to fix problematic K4s
        for _ in range(100):
            if time.time() - t_start > t_budget or time.time() > deadline:
                break
            # Find a K4 with minimum distinct colors
            min_dc = 6
            target_k4 = -1
            for k4_idx, e_indices in enumerate(k4s):
                dc = count_distinct_colors(e_indices, colors)
                if dc < min_dc:
                    min_dc = dc
                    target_k4 = k4_idx
                    if dc <= 3:
                        break
            
            if target_k4 == -1:
                break
                
            # Try to fix this K4 by changing one of its edges
            e_indices = k4s[target_k4]
            for ei in e_indices:
                old_color = colors[ei]
                for new_color in range(K):
                    if new_color == old_color:
                        continue
                    colors[ei] = new_color
                    new_penalty = calc_penalty(colors)
                    if new_penalty < current_penalty:
                        current_penalty = new_penalty
                        if current_penalty < best_penalty:
                            best_penalty = current_penalty
                            best_colors = colors[:]
                        if current_penalty == 0:
                            return colors
                        break
                    colors[ei] = old_color
            
            if current_penalty == 0:
                return best_colors
        
        # Simulated annealing phase
        T = 1.0
        while time.time() - t_start < t_budget and time.time() < deadline:
            # Pick random edge
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            new_color = random.randint(0, K - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate delta penalty efficiently
            old_delta = 0
            for k4_idx in edge_k4s[e_idx]:
                e_indices = k4s[k4_idx]
                old_dc = count_distinct_colors(e_indices, colors)
                old_delta += max(0, 5 - old_dc)
            
            colors[e_idx] = new_color
            new_delta = 0
            for k4_idx in edge_k4s[e_idx]:
                e_indices = k4s[k4_idx]
                new_dc = count_distinct_colors(e_indices, colors)
                new_delta += max(0, 5 - new_dc)
            
            delta_penalty = new_delta - old_delta
            new_penalty = current_penalty + delta_penalty
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta / T)
                if delta_penalty > 0:
                    prob = math.exp(-delta_penalty / T)
                    if random.random() < prob:
                        current_penalty = new_penalty
                    else:
                        # Revert
                        colors[e_idx] = old_color
                else:
                    # new_penalty == current_penalty (delta 0)
                    # Accept to explore
                    current_penalty = new_penalty
            
            if current_penalty == 0:
                return colors
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # If we found a better solution, update
        if best_penalty < current_penalty:
            colors = best_colors
            current_penalty = best_penalty
    
    # If no better solution found, return the 17-color solution
    # We should compress the 17-color solution to ensure indices are 0..16
    unique_colors = sorted(list(set(base_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [mapping[c] for c in base_colors]
    return final_colors
