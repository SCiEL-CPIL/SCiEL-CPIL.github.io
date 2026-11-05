"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    edge_idx_map = {e: i for i, e in enumerate(edges)}
    k4s = []
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        e_indices = [
            edge_idx_map[(v0, v1)],
            edge_idx_map[(v0, v2)],
            edge_idx_map[(v0, v3)],
            edge_idx_map[(v1, v2)],
            edge_idx_map[(v1, v3)],
            edge_idx_map[(v2, v3)]
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)  # 1820
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Fallback solution: (i+j) % 17
    seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    def count_colors(col):
        return len(set(col))
    
    def is_valid(col):
        for k in range(num_k4s):
            if len(set(col[e] for e in k4s[k])) < 5:
                return False
        return True

    best_result = None
    best_color_count = float('inf')
    
    # Check seed
    if is_valid(seed_col):
        best_result = seed_col[:]
        best_color_count = count_colors(seed_col)
        
    MAX_COLORS = 20
    
    def fast_local_search(initial_col, max_colors, max_iters, time_limit):
        col = initial_col[:]
        k4_color_counts = [[0] * MAX_COLORS for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        current_violations = 0
        
        for k in range(num_k4s):
            counts = [0] * MAX_COLORS
            for e in k4s[k]:
                c = col[e]
                if c < MAX_COLORS:
                    counts[c] += 1
            k4_color_counts[k] = counts
            d = sum(1 for x in counts if x > 0)
            k4_distinct[k] = d
            if d < 5:
                current_violations += 1
                
        if current_violations == 0:
            return col, count_colors(col)
            
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
                
            e_idx = random.randint(0, num_edges - 1)
            old_color = col[e_idx]
            new_color = random.randint(0, max_colors - 1)
            
            if old_color == new_color:
                continue
            
            if old_color >= MAX_COLORS or new_color >= MAX_COLORS:
                continue
                
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            col[e_idx] = new_color
            
            for k in affected_k4s:
                counts = k4_color_counts[k]
                old_d = k4_distinct[k]
                
                counts[old_color] -= 1
                counts[new_color] += 1
                
                new_d = sum(1 for x in counts if x > 0)
                k4_distinct[k] = new_d
                
                if old_d < 5 and new_d >= 5:
                    delta_viol -= 1
                elif old_d >= 5 and new_d < 5:
                    delta_viol += 1
                    
            if delta_viol <= 0:
                current_violations += delta_viol
            else:
                if random.random() < np.exp(-delta_viol / T):
                    current_violations += delta_viol
                else:
                    col[e_idx] = old_color
                    for k in affected_k4s:
                        counts = k4_color_counts[k]
                        counts[new_color] -= 1
                        counts[old_color] += 1
                        new_d = sum(1 for x in counts if x > 0)
                        k4_distinct[k] = new_d
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
            if current_violations == 0:
                return col, count_colors(col)
                
        return None, float('inf')

    def try_reduce_colors(col, max_colors, time_limit):
        """Try to reduce the number of colors used by merging color classes."""
        col = col[:]
        current_colors = count_colors(col)
        
        if current_colors <= 13:
            return col, current_colors
        
        # Try to eliminate one color at a time
        colors_used = sorted(set(col))
        
        for _ in range(20):  # Try up to 20 elimination attempts
            if time.time() > time_limit:
                break
            
            if len(colors_used) <= 13:
                break
            
            # Pick a color to try to eliminate (prefer less-used colors)
            color_counts = {}
            for c in colors_used:
                color_counts[c] = col.count(c)
            
            # Sort by count ascending - try to eliminate rare colors first
            sorted_colors = sorted(colors_used, key=lambda c: color_counts[c])
            target_color = sorted_colors[0]
            
            # Try to recolor all edges with target_color to other colors
            target_edges = [i for i in range(num_edges) if col[i] == target_color]
            
            if not target_edges:
                continue
            
            # For each edge with target_color, try to change it to a different color
            # that doesn't violate any K4 constraint
            success = True
            for e_idx in target_edges:
                found = False
                for new_color in colors_used:
                    if new_color == target_color:
                        continue
                    
                    # Check if changing this edge to new_color is safe
                    col[e_idx] = new_color
                    safe = True
                    for k in edge_to_k4s[e_idx]:
                        if len(set(col[e] for e in k4s[k])) < 5:
                            safe = False
                            break
                    
                    if safe:
                        found = True
                        break
                    else:
                        col[e_idx] = target_color
                
                if not found:
                    success = False
                    break
            
            if success:
                # Successfully eliminated target_color
                colors_used.remove(target_color)
                current_colors -= 1
                
                if current_colors <= 13:
                    return col, current_colors
        
        return col, count_colors(col)

    # Try to find better solutions
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 25.0:
            break
            
        for restart in range(80):
            if time.time() > deadline - 15.0:
                break
                
            # Random initialization
            col = [random.randint(0, target_colors - 1) for _ in range(num_edges)]
            
            result_col, c_count = fast_local_search(col, target_colors, 8000, deadline - 10.0)
            
            if result_col is not None:
                if c_count < best_color_count:
                    best_color_count = c_count
                    best_result = result_col[:]
                
                # Try to reduce colors further
                if c_count > 13 and time.time() < deadline - 10.0:
                    reduced_col, reduced_count = try_reduce_colors(result_col, target_colors, deadline - 8.0)
                    if reduced_count < best_color_count:
                        best_color_count = reduced_count
                        best_result = reduced_col[:]
                
                if c_count <= 13:
                    return best_result

    # Also try starting from seed and reducing
    if best_result is not None and best_color_count > 13 and time.time() < deadline - 15.0:
        reduced_col, reduced_count = try_reduce_colors(best_result, best_color_count, deadline - 5.0)
        if reduced_count < best_color_count:
            best_color_count = reduced_count
            best_result = reduced_col[:]

    if best_result is not None:
        return best_result
    
    return seed_col
# EVOLVE-BLOCK-END
