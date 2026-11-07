"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_cycles(coloring, n, edges):
    """Check if any pair of colors induces a subgraph with a cycle using union-find."""
    color_to_vertices = defaultdict(list)
    for v in range(n):
        color_to_vertices[coloring[v]].append(v)
    
    colors_used = sorted(color_to_vertices.keys())
    
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            ci, cj = colors_used[i], colors_used[j]
            vertices_in_pair = set(color_to_vertices[ci]) | set(color_to_vertices[cj])
            if len(vertices_in_pair) < 3:
                continue
            
            # Union-find
            parent = {v: v for v in vertices_in_pair}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            
            has_cycle = False
            for u, v in edges:
                if u in vertices_in_pair and v in vertices_in_pair:
                    if coloring[u] in (ci, cj) and coloring[v] in (ci, cj):
                        if not union(u, v):
                            has_cycle = True
                            break
            if has_cycle:
                return False
    
    return True

def _is_valid(coloring, n, edges):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return _check_bicolored_cycles(coloring, n, edges)

def _count_colors(coloring):
    return len(set(coloring))

def _greedy_coloring(n, edges, k, deadline):
    """Fast greedy coloring with constraint propagation."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    colors = [-1] * n
    # Order vertices by degree descending (all degree 4, so natural order)
    vertices = list(range(n))
    
    for v in vertices:
        if time.time() > deadline:
            return None
        used_colors = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                used_colors.add(colors[nb])
        
        # Try colors in order
        for c in range(k):
            if c not in used_colors:
                colors[v] = c
                break
        else:
            return None
    
    if _is_valid(colors, n, edges):
        return colors
    return None

def _local_search(n, edges, k, deadline):
    """Local search: start with greedy, then try to reduce colors."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Start with a valid coloring using k colors
    current = None
    for attempt in range(10):
        if time.time() > deadline:
            break
        random.seed(attempt)
        # Random permutation of vertices for greedy
        perm = list(range(n))
        random.shuffle(perm)
        
        colors = [-1] * n
        success = True
        for v in perm:
            used = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    used.add(colors[nb])
            for c in range(k):
                if c not in used:
                    colors[v] = c
                    break
            else:
                success = False
                break
        
        if success and _is_valid(colors, n, edges):
            current = colors[:]
            break
    
    if current is None:
        return None
    
    # Try to reduce colors by remapping
    best = current[:]
    best_count = _count_colors(best)
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        # Pick a random vertex and try to change its color to a smaller one
        v = random.randint(0, n - 1)
        old_c = best[v]
        new_c = random.randint(0, old_c - 1) if old_c > 0 else None
        
        if new_c is not None:
            # Check if new_c is available for v
            conflict = False
            for nb in adj[v]:
                if best[nb] == new_c:
                    conflict = True
                    break
            
            if not conflict:
                best[v] = new_c
                if _is_valid(best, n, edges):
                    new_count = _count_colors(best)
                    if new_count < best_count:
                        best_count = new_count
                else:
                    best[v] = old_c
    
    return best

def construct():
    n = P['n']
    edges = P['edges']
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, 40s safety margin
    
    best_coloring = list(range(n))  # Fallback: 50 colors
    best_count = n
    
    # Try k from 3 upward
    for k in range(3, min(n + 1, 15)):  # Cap at 15 to avoid excessive search
        if time.time() > deadline:
            break
        
        # Try greedy first
        greedy_result = _greedy_coloring(n, edges, k, deadline)
        if greedy_result is not None:
            cnt = _count_colors(greedy_result)
            if cnt < best_count:
                best_count = cnt
                best_coloring = greedy_result
                if best_count == 3:
                    break
        
        # Try local search
        if time.time() < deadline - 10:
            ls_result = _local_search(n, edges, k, deadline)
            if ls_result is not None:
                cnt = _count_colors(ls_result)
                if cnt < best_count:
                    best_count = cnt
                    best_coloring = ls_result
                    if best_count == 3:
                        break
        
        if best_count <= k:
            break
    
    # Final verification
    if not _is_valid(best_coloring, n, edges):
        best_coloring = list(range(n))
    
    result = [int(c) for c in best_coloring]
    return result
# EVOLVE-BLOCK-END
