import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute edge index map
    edge_idx_map = {e: i for i, e in enumerate(edges)}
    
    # Precompute K4s and their edge indices
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
    
    k4s = np.array(k4s, dtype=np.int32)
    num_k4s = len(k4s)
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    edge_to_k4s_np = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]
    
    def count_violations(col):
        """Count K4s with < 5 distinct colors."""
        violations = 0
        for i in range(num_k4s):
            k4_colors = col[k4s[i]]
            if len(set(k4_colors.tolist())) < 5:
                violations += 1
        return violations
    
    def initialize_caches(col):
        """Initialize caches for K4 distinct counts."""
        k4_distinct_cache = np.zeros(num_k4s, dtype=np.int32)
        for i in range(num_k4s):
            colors = col[k4s[i]]
            k4_distinct_cache[i] = len(set(colors.tolist()))
        return k4_distinct_cache
    
    def try_move(col, e_idx, new_color, k4_distinct_cache):
        """
        Try changing col[e_idx] to new_color.
        Returns (delta_violations, old_color, affected_k4s)
        """
        old_color = col[e_idx]
        if old_color == new_color:
            return 0, old_color, []
        
        affected_k4s = edge_to_k4s_np[e_idx]
        delta_viol = 0
        
        # Compute old distinct counts for affected K4s
        old_distincts = {}
        for k4_idx in affected_k4s:
            old_distincts[k4_idx] = k4_distinct_cache[k4_idx]
        
        # Apply the move
        col[e_idx] = new_color
        
        # Compute new distinct counts and delta
        for k4_idx in affected_k4s:
            new_colors = col[k4s[k4_idx]]
            new_distinct = len(set(new_colors.tolist()))
            old_distinct = old_distincts[k4_idx]
            
            old_is_valid = old_distinct >= 5
            new_is_valid = new_distinct >= 5
            
            if old_is_valid and not new_is_valid:
                delta_viol += 1
            elif not old_is_valid and new_is_valid:
                delta_viol -= 1
            
            k4_distinct_cache[k4_idx] = new_distinct
        
        return delta_viol, old_color, affected_k4s
    
    def revert_move(col, e_idx, old_color, affected_k4s, k4_distinct_cache):
        """Revert a move."""
        col[e_idx] = old_color
        for k4_idx in affected_k4s:
            colors = col[k4s[k4_idx]]
            k4_distinct_cache[k4_idx] = len(set(colors.tolist()))
    
    def local_search(col, max_iters=50000, time_limit=10.0, max_colors=15):
        """Local search to fix violations."""
        k4_distinct_cache = initialize_caches(col)
        current_violations = np.sum(k4_distinct_cache < 5)
        
        if current_violations == 0:
            return col, 0
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        start = time.time()
        for _ in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            e_idx = random.randint(0, num_edges - 1)
            new_color = random.randint(0, max_colors - 1)
            
            delta_viol, old_color, affected = try_move(col, e_idx, new_color, k4_distinct_cache)
            
            if delta_viol <= 0:
                current_violations += delta_viol
            else:
                if random.random() < np.exp(-delta_viol / T):
                    current_violations += delta_viol
                else:
                    revert_move(col, e_idx, old_color, affected, k4_distinct_cache)
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            if current_violations == 0:
                break
        
        return col, current_violations
    
    def eliminate_color(col, target_color, k4_distinct_cache, max_colors=15):
        """Try to eliminate a specific color. Returns True if successful."""
        edges_to_change = np.where(col == target_color)[0]
        
        for e_idx in edges_to_change:
            old_color = col[e_idx]
            found_new_color = False
            
            # Try all other colors
            for new_color in range(max_colors):
                if new_color == old_color:
                    continue
                
                delta_viol, old_c, affected = try_move(col, e_idx, new_color, k4_distinct_cache)
                
                if delta_viol <= 0:
                    found_new_color = True
                    break
                else:
                    revert_move(col, e_idx, old_c, affected, k4_distinct_cache)
            
            if not found_new_color:
                return False
        
        return True
    
    # Start with the known-valid algebraic coloring: (i+j) % 17
    # This uses 17 colors but is guaranteed valid
    seed_col = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    best_col = seed_col.copy()
    best_colors_used = len(set(seed_col.tolist()))
    
    # Try to reduce color count from 15 down to 13
    for target_colors in [15, 14, 13]:
        if time.time() > deadline - 10:
            break
        
        if best_colors_used <= target_colors:
            continue
        
        # Multiple restarts trying to find a valid coloring with target_colors colors
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start from the seed coloring and try to merge colors
            col = seed_col.copy()
            
            # First, remap colors to use only target_colors colors
            # Group similar colors and merge them
            unique_colors = np.unique(col)
            if len(unique_colors) > target_colors:
                # Create a mapping from old colors to new colors (0..target_colors-1)
                color_map = {}
                for i, c in enumerate(unique_colors):
                    color_map[c] = i % target_colors
                col = np.array([color_map[c] for c in col], dtype=np.int32)
            
            # Run local search to fix any violations
            col, violations = local_search(col, max_iters=30000, time_limit=12.0, max_colors=target_colors)
            
            if violations == 0:
                colors_in_use = len(set(col.tolist()))
                if colors_in_use < best_colors_used:
                    best_col = col.copy()
                    best_colors_used = colors_in_use
                
                # Try to eliminate more colors
                k4_distinct_cache = initialize_caches(col)
                
                while best_colors_used > 13 and time.time() < deadline - 3:
                    colors_in_use = set(col.tolist())
                    
                    if len(colors_in_use) <= 13:
                        break
                    
                    # Pick color with fewest edges to eliminate
                    color_counts = {}
                    for c in colors_in_use:
                        color_counts[c] = np.sum(col == c)
                    
                    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
                    
                    eliminated = False
                    for target_color, count in sorted_colors:
                        if count == 0:
                            continue
                        
                        col_copy = col.copy()
                        cache_copy = k4_distinct_cache.copy()
                        
                        if eliminate_color(col_copy, target_color, cache_copy, max_colors=target_colors):
                            new_violations = count_violations(col_copy)
                            if new_violations == 0:
                                col = col_copy
                                k4_distinct_cache = cache_copy
                                best_col = col.copy()
                                best_colors_used = len(set(col.tolist()))
                                eliminated = True
                                break
                    
                    if not eliminated:
                        break
                
                if best_colors_used <= target_colors:
                    break
        
        if best_colors_used <= target_colors:
            break
    
    # Final check: if best is still 17, try harder with random restarts
    if best_colors_used > 15:
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            
            col = np.random.randint(0, 15, size=num_edges)
            col, violations = local_search(col, max_iters=20000, time_limit=10.0, max_colors=15)
            
            if violations == 0:
                colors_in_use = len(set(col.tolist()))
                if colors_in_use < best_colors_used:
                    best_col = col.copy()
                    best_colors_used = colors_in_use
    
    return best_col.tolist()
