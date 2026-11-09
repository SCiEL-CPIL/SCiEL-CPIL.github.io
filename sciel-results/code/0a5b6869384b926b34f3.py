import time
import random
import numpy as np
from itertools import combinations, permutations

def get_edge_index(i, j, n):
    if i > j:
        i, j = j, i
    return sum(n - 1 - k for k in range(i)) + (j - i - 1)

def precompute_cycles(n):
    """Precompute all 6-cycles as numpy array of edge indices."""
    base_cycles = set()
    for perm in permutations([1, 2, 3, 4, 5]):
        cycle = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        rev = (0, perm[4], perm[3], perm[2], perm[1], perm[0])
        canon = min(cycle, rev)
        base_cycles.add(canon)
    
    cycle_edges = []
    for comb in combinations(range(n), 6):
        mapping = {k: v for k, v in enumerate(comb)}
        for b_cycle in base_cycles:
            mapped_cycle = tuple(mapping[v] for v in b_cycle)
            edges = []
            for k in range(6):
                u = mapped_cycle[k]
                v = mapped_cycle[(k + 1) % 6]
                edges.append(get_edge_index(u, v, n))
            cycle_edges.append(edges)
    
    return np.array(cycle_edges, dtype=np.int32)

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    
    n = 17
    num_edges = n * (n - 1) // 2
    cycles = precompute_cycles(n)
    num_cycles = len(cycles)
    
    # Penalty function using NumPy: count monochromatic 6-cycles
    # colors: numpy array of shape (num_edges,) with values 0-3
    def penalty(colors):
        # For each cycle, check if all 6 edges have the same color
        # colors[cycles] gives shape (num_cycles, 6)
        cycle_colors = colors[cycles]  # shape (num_cycles, 6)
        # A cycle is monochromatic if all 6 values are equal
        # Check: max - min == 0 for each row
        return int(np.sum(np.max(cycle_colors, axis=1) - np.min(cycle_colors, axis=1) == 0))
    
    # Local search with simulated annealing
    best_penalty = float('inf')
    best_colors = None
    
    def time_remaining():
        return deadline - time.time()
    
    for attempt in range(200):
        if time_remaining() < 5.0:
            break
            
        # Random initialization
        colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
        cur_penalty = penalty(colors)
        
        if cur_penalty < best_penalty:
            best_penalty = cur_penalty
            best_colors = colors.copy()
            
        if cur_penalty == 0:
            break
            
        # Simulated annealing
        T = 10.0
        cooling = 0.95
        max_steps = 500
        stuck_count = 0
        
        for step in range(max_steps):
            if time_remaining() < 2.0:
                    break
            if stuck_count > 200:
                break
                
            # Pick a random edge to change
            idx = random.randint(0, num_edges - 1)
            old_color = colors[idx]
            
            # Find the color that minimizes penalty
            best_c = old_color
            min_p = cur_penalty
            
            for c in range(4):
                if c == old_color:
                    continue
                colors[idx] = c
                new_p = penalty(colors)
                if new_p < min_p:
                    min_p = new_p
                    best_c = c
            colors[idx] = best_c
            
            if min_p < cur_penalty:
                cur_penalty = min_p
                stuck_count = 0
                if cur_penalty < best_penalty:
                    best_penalty = cur_penalty
                    best_colors = colors.copy()
                if cur_penalty == 0:
                    break
            else:
                # Accept worse move with probability exp(-delta/T)
                delta = min_p - cur_penalty
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        cur_penalty = min_p
                        stuck_count = 0
                    else:
                        colors[idx] = old_color
                        stuck_count += 1
                else:
                    colors[idx] = old_color
                    stuck_count += 1
            
            T *= cooling
        
        if cur_penalty == 0:
            break
            
    if best_penalty == 0:
        return {'n': n, 'colors': best_colors.tolist()}
        
    # Fallback: try smaller n
    for n_fallback in [16, 15, 14, 13, 12, 11, 10, 9]:
        if time_remaining() < 1.0:
            break
        num_edges_f = n_fallback * (n_fallback - 1) // 2
        cycles_f = precompute_cycles(n_fallback)
        
        for attempt in range(100):
            if time_remaining() < 1.0:
                break
            colors_f = np.random.randint(0, 4, size=num_edges_f, dtype=np.int8)
            cur_p = penalty(colors_f)
            if cur_p < best_penalty:
                best_penalty = cur_p
                best_colors = colors_f.copy()
            if cur_p == 0:
                break
            # Simple hill climbing
            for step in range(100):
                if time_remaining() < 1.0:
                    break
                idx = random.randint(0, num_edges_f - 1)
                old_c = colors_f[idx]
                best_c = old_c
                min_p = cur_p
                for c in range(4):
                    if c == old_c:
                        continue
                    colors_f[idx] = c
                    new_p = penalty(colors_f)
                    if new_p < min_p:
                        min_p = new_p
                        best_c = c
                colors_f[idx] = best_c
                if min_p < cur_p:
                    cur_p = min_p
                    if cur_p < best_penalty:
                        best_penalty = cur_p
                        best_colors = colors_f.copy()
                    if cur_p == 0:
                        break
                else:
                    colors_f[idx] = old_c
            if cur_p == 0:
                break
        if best_penalty == 0:
            return {'n': n_fallback, 'colors': best_colors.tolist()}
    
    # Final fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
