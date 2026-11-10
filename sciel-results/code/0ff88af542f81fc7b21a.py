"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def full_check(colors):
        """Full validity check: adjacency + bicolored forest for all color pairs."""
        # Rule 1: adjacent vertices must differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Rule 2: for every pair of colors, induced subgraph is a forest
        color_to_verts = defaultdict(set)
        for i, c in enumerate(colors):
            color_to_verts[c].add(i)
        
        used_colors = sorted(color_to_verts.keys())
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                verts_set = color_to_verts[c1] | color_to_verts[c2]
                if len(verts_set) <= 1:
                    continue
                # Check for cycles using union-find
                parent = {v: v for v in verts_set}
                rank = {v: 0 for v in verts_set}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                for u, v in edges:
                    if u in verts_set and v in verts_set:
                        if not union(u, v):
                            return False
        return True
    
    def try_color_change(v, new_c, colors):
        """
        Try to change vertex v's color to new_c.
        Returns True if the change maintains validity.
        """
        old_c = colors[v]
        if new_c == old_c:
            return True
        
        # Check adjacency: no neighbor has new_c
        for u in adj[v]:
            if colors[u] == new_c:
                return False
        
        # Full check on modified colors
        test_colors = colors[:]
        test_colors[v] = new_c
        return full_check(test_colors)
    
    # Strategy: try k from 2 upward, use local search with multiple restarts
    best_result = None
    
    for k in range(2, 10):
        if time.time() > deadline:
            break
        
        # Multiple restarts with different initializations
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Modular by k
                colors = [i % k for i in range(n)]
            elif restart == 1:
                # Modular by k with offset
                colors = [(i + 1) % k for i in range(n)]
            elif restart == 2:
                # Modular by k with different stride
                colors = [(i * 2) % k for i in range(n)]
            elif restart == 3:
                # Two-phase pattern
                half = n // 2
                colors = [i % k for i in range(half)] + [(i % k + k // 2) % k for i in range(half, n)]
            else:
                # Random initialization
                colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Fix adjacency violations greedily
            for _ in range(50):
                changed = False
                for v in range(n):
                    conflict = False
                    for u in adj[v]:
                        if colors[u] == colors[v]:
                            conflict = True
                            break
                    if conflict:
                        for c in range(k):
                            if c == colors[v]:
                                continue
                            ok = True
                            for u in adj[v]:
                                if colors[u] == c:
                                    ok = False
                                    break
                            if ok:
                                colors[v] = c
                                changed = True
                                break
                        if not changed:
                            colors[v] = random.randint(0, k - 1)
                            changed = True
                if not changed:
                    break
            
            # Check if current coloring is valid
            if full_check(colors):
                distinct = len(set(colors))
                if best_result is None or distinct < len(set(best_result)):
                    best_result = colors[:]
            
            # Local search: try to reduce number of distinct colors
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                current_distinct = len(set(colors))
                if current_distinct <= k and best_result is not None and current_distinct >= len(set(best_result)):
                    # No improvement possible within this k
                    break
                
                # Count colors
                color_counts = defaultdict(int)
                for c in colors:
                    color_counts[c] += 1
                
                # Try to eliminate rare colors
                sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
                
                improved = False
                for target_color in sorted_colors:
                    if time.time() > deadline:
                        break
                    if color_counts[target_color] == 0:
                        continue
                    
                    vertices_with_color = [v for v in range(n) if colors[v] == target_color]
                    random.shuffle(vertices_with_color)
                    
                    for v in vertices_with_color:
                        # Try changing v to each other color
                        for new_c in range(k):
                            if new_c == target_color:
                                continue
                            if try_color_change(v, new_c, colors):
                                colors[v] = new_c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    # Stuck, try random perturbation
                    if random.random() < 0.5:
                        v = random.randint(0, n - 1)
                        new_c = random.randint(0, k - 1)
                        if new_c != colors[v]:
                            if try_color_change(v, new_c, colors):
                                colors[v] = new_c
                    else:
                        break
            
            # Final check
            if full_check(colors):
                distinct = len(set(colors))
                if best_result is None or distinct < len(set(best_result)):
                    best_result = colors[:]
        
        # If we found a valid coloring with exactly k colors, we can stop
        if best_result is not None and len(set(best_result)) == k:
            break
    
    # Fallback: return all distinct colors
    if best_result is None:
        return [int(i) for i in range(n)]
    
    # Remap colors to 0..distinct-1
    distinct = len(set(best_result))
    color_map = {}
    next_id = 0
    remapped = []
    for c in best_result:
        if c not in color_map:
            color_map[c] = next_id
            next_id += 1
        remapped.append(color_map[c])
    return [int(c) for c in remapped]

# EVOLVE-BLOCK-END
