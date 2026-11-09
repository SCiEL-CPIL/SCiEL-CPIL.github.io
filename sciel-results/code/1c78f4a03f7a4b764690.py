import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute K4s and edge mappings
    # k4s: list of 4-tuples of vertices
    # edge_to_k4s: for each edge index, list of (k4_index, position_in_k4)
    
    # Map (u, v) to edge index
    # u < v
    edge_index = {}
    edge_vertices = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = idx
            edge_vertices.append((u, v))
            idx += 1
            
    k4s = []
    k4_edges = [] # list of lists of edge indices
    k4_idx_map = {} # (u,v) -> list of (k4_idx, pos)
    
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                for x in range(w + 1, n):
                    k4s.append((u, v, w, x))
                    # Edges: (u,v), (u,w), (u,x), (v,w), (v,x), (w,x)
                    e1 = edge_index[(u, v)]
                    e2 = edge_index[(u, w)]
                    e3 = edge_index[(u, x)]
                    e4 = edge_index[(v, w)]
                    e5 = edge_index[(v, x)]
                    e6 = edge_index[(w, x)]
                    
                    k4_idx = len(k4_edges)
                    k4_edges.append([e1, e2, e3, e4, e5, e6])
                    
                    # Update reverse mapping
                    if e1 not in k4_idx_map: k4_idx_map[e1] = []
                    k4_idx_map[e1].append((k4_idx, 0))
                    if e2 not in k4_idx_map: k4_idx_map[e2] = []
                    k4_idx_map[e2].append((k4_idx, 1))
                    if e3 not in k4_idx_map: k4_idx_map[e3] = []
                    k4_idx_map[e3].append((k4_idx, 2))
                    if e4 not in k4_idx_map: k4_idx_map[e4] = []
                    k4_idx_map[e4].append((k4_idx, 3))
                    if e5 not in k4_idx_map: k4_idx_map[e5] = []
                    k4_idx_map[e5].append((k4_idx, 4))
                    if e6 not in k4_idx_map: k4_idx_map[e6] = []
                    k4_idx_map[e6].append((k4_idx, 5))
                    
    num_k4s = len(k4_edges)
    
    # Helper to calculate badness for a given coloring
    # coloring: np array of length num_edges
    def calc_badness(coloring):
        # k4_colors: (num_k4s, 6)
        k4_colors = np.array([coloring[e] for e in k4_edges])
        # Count distinct colors in each row
        # Sort each row, then count changes
        # Or use np.unique on each row? Slow.
        # Better: for each row, sort, then count diffs > 0.
        sorted_colors = np.sort(k4_colors, axis=1)
        # distinct count = 1 + sum(sorted_colors[:, 1:] != sorted_colors[:, :-1])
        diffs = sorted_colors[:, 1:] != sorted_colors[:, :-1]
        distinct_counts = 1 + diffs.sum(axis=1)
        # Badness: max(0, 5 - distinct_count)
        badness = np.maximum(0, 5 - distinct_counts)
        return badness.sum()

    # Local Search
    def local_search(K, max_iters=50000, seed=42):
        random.seed(seed)
        # Initialize with random colors in 0..K-1
        coloring = np.array([random.randint(0, K - 1) for _ in range(num_edges)], dtype=np.int32)
        current_badness = calc_badness(coloring)
        
        if current_badness == 0:
            # Check actual colors used
            actual_colors = len(np.unique(coloring))
            return coloring, actual_colors
        
        best_coloring = coloring.copy()
        best_badness = current_badness
        best_colors = len(np.unique(coloring))
        
        temperature = 1.0
        cooling = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            old_c = coloring[e]
            # Pick a new color
            if K == 1:
                new_c = old_c
            else:
                new_c = random.randint(0, K - 1)
                if new_c == old_c:
                    continue
            
            # Calculate delta badness
            # Only K4s containing e are affected
            affected = k4_idx_map[e]
            
            delta_badness = 0
            # We need to update the k4_colors temporarily or calculate the change
            # Since we don't have a persistent k4_colors array in this scope for speed, 
            # we can just recalculate badness for affected K4s.
            # But calc_badness is global. 
            # Let's maintain a persistent k4_colors array for speed.
            
            # Actually, implementing a persistent k4_colors array is better.
            # Let's refactor local_search to use persistent arrays.
            pass
            
        return best_coloring, best_colors

    # Refactored Local Search with Persistent State
    def local_search_optimized(K, max_iters=200000, seed=42):
        random.seed(seed)
        coloring = np.array([random.randint(0, K - 1) for _ in range(num_edges)], dtype=np.int32)
        
        # Persistent k4_colors
        k4_colors = np.array([coloring[e] for e in k4_edges], dtype=np.int32)
        
        # Calculate initial badness
        sorted_colors = np.sort(k4_colors, axis=1)
        diffs = sorted_colors[:, 1:] != sorted_colors[:, :-1]
        distinct_counts = 1 + diffs.sum(axis=1)
        badness_per_k4 = np.maximum(0, 5 - distinct_counts)
        current_badness = badness_per_k4.sum()
        
        if current_badness == 0:
            return coloring, len(np.unique(coloring))
            
        best_coloring = coloring.copy()
        best_badness = current_badness
        best_colors = len(np.unique(coloring))
        
        # Simple Hill Climbing with Random Restarts
        # We'll just do a long run of random moves, accepting if better or equal (to escape plateaus)
        # Or Simulated Annealing.
        
        temp = 1.0
        cool = 0.9995
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            e = random.randint(0, num_edges - 1)
            old_c = coloring[e]
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
            
            # Update affected K4s
            affected = k4_idx_map[e]
            
            delta_badness = 0
            # Store old badness for affected to restore if rejected
            old_badness_values = []
            for k4_idx, pos in affected:
                old_bad = badness_per_k4[k4_idx]
                old_badness_values.append(old_bad)
                
                # Update k4_colors
                k4_colors[k4_idx, pos] = new_c
                # Recalculate distinct count for this K4
                # Extract row, sort, count
                row = k4_colors[k4_idx]
                s = np.sort(row)
                d = s[1:] != s[:-1]
                distinct = 1 + d.sum()
                new_bad = max(0, 5 - distinct)
                
                delta_badness += new_bad - old_bad
                badness_per_k4[k4_idx] = new_bad
            
            new_badness = current_badness + delta_badness
            
            # Acceptance criterion
            if new_badness < current_badness or (new_badness == current_badness and random.random() < 0.1):
                # Accept
                coloring[e] = new_c
                current_badness = new_badness
                
                if current_badness < best_badness:
                    best_badness = current_badness
                    best_coloring = coloring.copy()
                    best_colors = len(np.unique(coloring))
                    
                    if best_badness == 0:
                        return best_coloring, best_colors
                        
            else:
                # Reject: restore state
                coloring[e] = old_c # already old_c
                for j, (k4_idx, pos) in enumerate(affected):
                    k4_colors[k4_idx, pos] = old_c
                    badness_per_k4[k4_idx] = old_badness_values[j]
                    
            temp *= cool
            if temp < 0.01:
                temp = 0.01

        return best_coloring, best_colors

    # Try K from 13 to 16
    for K in [13, 14, 15, 16]:
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts
        found_valid = False
        best_colors_for_K = K
        best_valid_coloring = None
        
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            coloring, colors_used = local_search_optimized(K, max_iters=100000, seed=restart)
            if colors_used <= K:
                # Verify validity just in case
                # The function returns best_coloring which had badness 0?
                # No, it returns best_coloring which had min badness.
                # We need to check if badness was 0.
                # Let's re-verify
                bad = calc_badness(coloring)
                if bad == 0:
                    if colors_used < best_colors_for_K:
                        best_colors_for_K = colors_used
                        best_valid_coloring = coloring
                    found_valid = True
                    break # Found valid for this K, move to next K (lower)
        
        if found_valid:
            # Return the coloring with min colors
            # Ensure the output format is correct: list of ints
            return list(best_valid_coloring)
            
    # If no valid coloring found with < 17 colors, return seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# EVOLVE-BLOCK-END
