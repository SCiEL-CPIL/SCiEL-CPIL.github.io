"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolor_cycle(adj, color, a, b):
    """Check if subgraph induced by colors a and b contains a cycle using union-find."""
    n = P['n']
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return True  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False
    
    # Collect edges where both endpoints have colors in {a, b}
    for u in range(n):
        if color[u] not in (a, b):
            continue
        for v in adj[u]:
            if v > u and color[v] in (a, b):
                if union(u, v):
                    return True
    return False

def _is_valid(color, adj):
    """Check all constraints: proper coloring and no bicolored cycles."""
    n = P['n']
    # Check proper coloring
    for u in range(n):
        for v in adj[u]:
            if v > u and color[u] == color[v]:
                return False
    # Check bicolored cycles for each pair of colors used
    colors_used = set(color)
    colors_list = list(colors_used)
    for i in range(len(colors_list)):
        for j in range(i + 1, len(colors_list)):
            if _has_bicolor_cycle(adj, color, colors_list[i], colors_list[j]):
                return False
    return True

def construct():
    deadline = time.time() + 80.0  # 80s safety margin
    n = P['n']
    adj = _build_adj()
    
    # Try with increasing number of colors
    for max_colors in range(2, 50):
        if time.time() > deadline:
            break
        best = None
        # Backtracking search
        color = [-1] * n
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            v = order[idx]
            # Try colors in order of least usage (or just 0..max_colors-1)
            for c in range(max_colors):
                # Check proper coloring with neighbors
                ok = True
                for u in adj[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                color[v] = c
                # Check bicolored cycles involving color c
                # Only need to check pairs (c, d) where d is already used
                cycle_found = False
                for d in range(c):
                    if any(color[x] == d for x in range(n)):
                        if _has_bicolor_cycle(adj, color, c, d):
                            cycle_found = True
                            break
                if not cycle_found:
                    if backtrack(idx + 1):
                        return True
                color[v] = -1
            return False
        
        if backtrack(0):
            # Verify the solution
            if _is_valid(color, adj):
                return [int(c) for c in color]
            else:
                # Should not happen, but just in case
                pass
    
    # Fallback: return a valid (but not optimal) coloring
    # Use greedy with many colors
    color = [0] * n
    for v in range(n):
        used = set(color[u] for u in adj[v])
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return [int(c) for c in color]

# EVOLVE-BLOCK-END
