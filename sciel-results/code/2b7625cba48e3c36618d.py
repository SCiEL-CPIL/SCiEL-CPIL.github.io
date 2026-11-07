"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _has_bicolored_cycle(coloring, edges):
    """Check if any pair of colors induces a cycle using union-find."""
    n = len(coloring)
    # Group vertices by color
    color_to_verts = defaultdict(list)
    for v, c in enumerate(coloring):
        color_to_verts[c].append(v)
    
    colors = list(color_to_verts.keys())
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            ci, cj = colors[i], colors[j]
            # Build union-find for vertices of color ci or cj
            verts = set(color_to_verts[ci]) | set(color_to_verts[cj])
            if len(verts) < 3:
                continue
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Check edges between these two color sets
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == ci and cv == cj) or (cu == cj and cv == ci):
                    if not union(u, v):
                        return True  # Found a cycle
    return False

def _is_valid(coloring, edges):
    """Check proper coloring and no bicolored cycles."""
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    if _has_bicolored_cycle(coloring, edges):
        return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def _greedy_coloring(n, edges, k):
    """Try to color with at most k colors, greedy."""
    coloring = [-1] * n
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for v in range(n):
        used = set(coloring[u] for u in adj[v] if coloring[u] >= 0)
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
        else:
            return None
    return coloring

def _backtrack_color(n, edges, k, deadline):
    """Backtracking search for valid coloring with exactly k colors."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    coloring = [-1] * n
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    order_idx = {v: i for i, v in enumerate(order)}
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            if _is_valid(coloring, edges):
                return coloring[:]
            return None
        
        v = order[idx]
        used = set(coloring[u] for u in adj[v] if coloring[u] >= 0)
        for c in range(k):
            if c in used:
                continue
            coloring[v] = c
            # Quick check: for each neighbor already colored, check if creating a bicolored cycle
            # This is expensive, so we do full check at the end
            result = backtrack(idx + 1)
            if result is not None:
                return result
        coloring[v] = -1
        return None
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    edges = P['edges']
    n = P['n']
    
    # Try increasing number of colors
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
        coloring = _backtrack_color(n, edges, k, deadline - 5)
        if coloring is not None and _is_valid(coloring, edges):
            return [int(c) for c in coloring]
    
    # Fallback: greedy with many colors
    coloring = _greedy_coloring(n, edges, 50)
    if coloring is None:
        coloring = list(range(n))
    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END

