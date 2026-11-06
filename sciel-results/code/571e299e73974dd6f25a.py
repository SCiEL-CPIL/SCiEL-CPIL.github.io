import random
import sys
import time

N = 16
sys.setrecursionlimit(10000)

def construct():
    edges = []
    edge_index = {}
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_index[(i, j)] = len(edges) - 1
            edge_index[(j, i)] = len(edges) - 1
            
    num_edges = len(edges)
    
    # Precompute paths for each edge
    # paths_for_edge[e_idx] = list of (w, x) such that u-w-x-v is a path
    paths_for_edge = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(edges):
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                paths_for_edge[idx].append((w, x))

    best_count = 0
    best_assignment = [0] * num_edges
    
    # State
    colors = [-1] * num_edges
    
    def get_valid_colors(idx):
        u, v = edges[idx]
        valid_mask = (1 << 120) - 1 # All colors valid initially
        
        for w, x in paths_for_edge[idx]:
            # Get colors of edges (u,w), (w,x), (x,v)
            c1 = colors[edge_index[(u, w)]]
            c2 = colors[edge_index[(w, x)]]
            c3 = colors[edge_index[(x, v)]]
            
            if c1 == -1 or c2 == -1 or c3 == -1:
                continue
                
            # If 3 distinct, the new color must be one of them
            if c1 != c2 and c2 != c3 and c1 != c3:
                # Mask for {c1, c2, c3}
                mask = (1 << c1) | (1 << c2) | (1 << c3)
                valid_mask &= mask
        
        return valid_mask

    def dfs(idx, used_colors_set):
        nonlocal best_count, best_assignment
        
        # Pruning: Even if we use a new color for every remaining edge, can we beat best_count?
        remaining = num_edges - idx
        max_possible = len(used_colors_set) + remaining
        if max_possible <= best_count:
            return False
            
        if idx == num_edges:
            if len(used_colors_set) > best_count:
                best_count = len(used_colors_set)
                best_assignment = colors[:]
            return True

        valid_mask = get_valid_colors(idx)
        
        if valid_mask == 0:
            return False

        # Extract valid colors
        new_colors = []
        old_colors = []
        
        for c in range(120):
            if valid_mask & (1 << c):
                if c in used_colors_set:
                    old_colors.append(c)
                else:
                    new_colors.append(c)
                    
        random.shuffle(new_colors)
        random.shuffle(old_colors)
        
        for c in new_colors:
            colors[idx] = c
            used_colors_set.add(c)
            
            if dfs(idx + 1, used_colors_set):
                return True
            
            colors[idx] = -1
            used_colors_set.remove(c)
            
        for c in old_colors:
            colors[idx] = c
            
            if dfs(idx + 1, used_colors_set):
                return True
            
            colors[idx] = -1
            
        return False

    start_time = time.time()
    TIME_LIMIT = 55 # seconds
    
    for attempt in range(20):
        random.seed(attempt)
        colors = [-1] * num_edges
        used_colors_set = set()
        
        dfs(0, used_colors_set)
        
        if best_count >= 110: # High enough?
            break
            
        if time.time() - start_time > TIME_LIMIT:
            break

    return best_assignment
