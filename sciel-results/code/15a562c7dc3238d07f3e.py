"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(colors, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest using union-find."""
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[colors[i]].append(i)
    
    colors_used = sorted(color_groups.keys())
    
    for idx_c1 in range(len(colors_used)):
        for idx_c2 in range(idx_c1 + 1, len(colors_used)):
            c1, c2 = colors_used[idx_c1], colors_used[idx_c2]
            vertices = set(color_groups[c1]) | set(color_groups[c2])
            if len(vertices) < 2:
                continue
            
            parent = {v: v for v in vertices}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            
            for u in vertices:
                for v in adj[u]:
                    if v in vertices and u < v:
                        if not union(u, v):
                            return False
    return True

def _is_valid_coloring(colors, adj, n):
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    return _check_bicolor_forest(colors, adj, n)

def _count_distinct(colors):
    return len(set(colors))

def _fast_check_pair(c1, c2, colors, adj, n):
    """Check if subgraph induced by colors c1 and c2 is a forest."""
    verts = set()
    for i in range(n):
        if colors[i] == c1 or colors[i] == c2:
            verts.add(i)
    if len(verts) < 2:
        return True
    parent = {v: v for v in verts}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u in verts:
        for v in adj[u]:
            if v in verts and u < v:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False
                parent[ru] = rv
    return True

def _check_after_change(v, old_c, new_c, colors, adj, n):
    """After changing vertex v from old_c to new_c, check all affected color pairs."""
    # Pairs to check: (new_c, c) for all c != new_c, and (old_c, c) for all c != old_c
    # But we only need to check pairs where the structure might have changed
    # Actually, adding v to new_c group and removing from old_c group affects:
    # - pairs (new_c, x) for all x != new_c: v is now in the induced subgraph
    # - pairs (old_c, x) for all x != old_c: v is removed from the induced subgraph
    # Removing a vertex can't create a cycle, so we only need to check (new_c, x) pairs
    all_colors = set(colors)
    for x in all_colors:
        if x == new_c:
            continue
        if not _fast_check_pair(new_c, x, colors, adj, n):
            return False
    return True

def _backtrack_fast(k, adj, n, deadline):
    """Fast backtracking with incremental checks for k-coloring."""
    colors = [-1] * n
    order = list(range(n))
    
    # Use a smarter ordering: try to color in a way that maximizes constraints
    # For a grid, row-by-row works well
    n_rows, n_cols = 5, 10
    order = [r * n_cols + c for r in range(n_rows) for c in range(n_cols)]
    
    nodes_visited = [0]
    
    def bt(idx):
        if time.time() > deadline:
            return False
        nodes_visited[0] += 1
        if idx == n:
            return True
        v = order[idx]
        for c in range(k):
            # Check proper coloring with already-colored neighbors
            valid = True
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[v] = c
            
            # Check bicolored forest constraint for pairs involving c
            ok = True
            for c2 in range(k):
                if c2 == c:
                    continue
                if not _fast_check_pair(c, c2, colors, adj, n):
                    ok = False
                    break
            
            if ok:
                if bt(idx + 1):
                    return True
            colors[v] = -1
        return False
    
    if bt(0):
        return colors[:]
    return None

def _local_search_reduce(colors, adj, n, deadline):
    """Local search to reduce number of distinct colors."""
    current = colors[:]
    distinct = len(set(current))
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            old_c = current[v]
            used_colors = sorted(set(current))
            for tc in used_colors:
                if tc == old_c:
                    continue
                # Check proper coloring
                proper = True
                for u in adj[v]:
                    if current[u] == tc:
                        proper = False
                        break
                if not proper:
                    continue
                
                current[v] = tc
                if _check_after_change(v, old_c, tc, current, adj, n):
                    new_d = len(set(current))
                    if new_d < distinct:
                        distinct = new_d
                        improved = True
                        break
                    else:
                        current[v] = old_c
                else:
                    current[v] = old_c
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Strategy 1: Fast backtracking for k=3, 4
    for k in [3, 4]:
        if time.time() > deadline:
            break
        result = _backtrack_fast(k, adj, n, deadline)
        if result is not None:
            score = _count_distinct(result)
            if score < best_score:
                best_score = score
                best_result = result[:]
            if best_score <= 3:
                break
    
    # Strategy 2: Pattern-based + local search
    if best_score > 4:
        patterns = []
        
        def pat1(i):
            row, col = divmod(i, 10)
            return (row + col) % 4
        patterns.append(pat1)
        
        def pat2(i):
            row, col = divmod(i, 10)
            return (row * 2 + col) % 4
        patterns.append(pat2)
        
        def pat3(i):
            row, col = divmod(i, 10)
            return (row + 2 * col) % 4
        patterns.append(pat3)
        
        def pat4(i):
            return i % 4
        patterns.append(pat4)
        
        def pat5(i):
            row, col = divmod(i, 10)
            return (row + col) % 5
        patterns.append(pat5)
        
        def pat6(i):
            row, col = divmod(i, 10)
            return (row * 3 + col) % 5
        patterns.append(pat6)
        
        for pat in patterns:
            if time.time() > deadline:
                break
            
            colors = [pat(i) for i in range(n)]
            
            if not _is_valid_coloring(colors, adj, n):
                continue
            
            score = _count_distinct(colors)
            if score < best_score:
                best_score = score
                best_result = colors[:]
            
            # Try local search reduction
            if best_score > 3:
                reduced = _local_search_reduce(colors, adj, n, deadline)
                r_score = _count_distinct(reduced)
                if r_score < best_score:
                    best_score = r_score
                    best_result = reduced[:]
            
            if best_score <= 3:
                break
    
    # Fallback: greedy coloring
    if best_result is None:
        colors = [0] * n
        for i in range(n):
            used = set()
            for v in adj[i]:
                if v < i:
                    used.add(colors[v])
            for c in range(10):
                if c not in used:
                    colors[i] = c
                    break
            else:
                colors[i] = max(used) + 1 if used else 0
        
        if _is_valid_coloring(colors, adj, n):
            best_result = colors
            best_score = _count_distinct(colors)
        else:
            best_result = list(range(n))
            best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
