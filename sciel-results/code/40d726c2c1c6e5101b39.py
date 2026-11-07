"""Solver for few-color graph coloring with no bicolored cycles."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_valid(coloring, adj):
    """Check adjacent differ and no bicolored cycles. Returns True if valid."""
    n = len(coloring)
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    color_pairs = set()
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        color_pairs.add((min(cu, cv), max(cu, cv)))
    for c1, c2 in color_pairs:
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True

def _merge_check(coloring, adj, color_a, color_b):
    """Check if merging color_b into color_a maintains validity."""
    n = len(coloring)
    # First check: no two adjacent vertices both get color_a after merge
    for u, v in P['edges']:
        new_u = color_a if coloring[u] == color_b else coloring[u]
        new_v = color_a if coloring[v] == color_b else coloring[v]
        if new_u == new_v:
            return False
    # Check bicolored cycles after merge
    new_coloring = [color_a if c == color_b else c for c in coloring]
    return _check_valid(new_coloring, adj)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    adj = _build_adj()
    n = P['n']
    
    # Start with a valid coloring using many colors, then merge
    # Use a pattern that's likely valid: v % 10
    coloring = [v % 10 for v in range(n)]
    
    # Verify it's valid
    if not _check_valid(coloring, adj):
        # Fallback to all distinct
        coloring = list(range(n))
    
    # Now try to merge colors to reduce count
    # Get list of distinct colors in use
    distinct_colors = sorted(set(coloring))
    
    # Try merging pairs of colors, starting with most frequent pairs
    from collections import Counter
    color_counts = Counter(coloring)
    
    # Repeatedly try to merge the two most common colors
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        distinct_colors = sorted(set(coloring))
        if len(distinct_colors) <= 2:
            break
        color_counts = Counter(coloring)
        # Sort by frequency descending
        sorted_colors = sorted(distinct_colors, key=lambda c: -color_counts[c])
        # Try merging pairs from most common
        for i in range(len(sorted_colors)):
            for j in range(i+1, len(sorted_colors)):
                ca, cb = sorted_colors[i], sorted_colors[j]
                # Quick check: are there adjacent vertices with these two colors?
                # If yes, merging would make them same color -> invalid
                has_adj = False
                for u, v in P['edges']:
                    cu, cv = coloring[u], coloring[v]
                    if (cu == ca and cv == cb) or (cu == cb and cv == ca):
                        has_adj = True
                        break
                if has_adj:
                    continue
                # Try the merge
                new_coloring = [ca if c == cb else c for c in coloring]
                if _check_valid(new_coloring, adj):
                    coloring = new_coloring
                    changed = True
                    break
            if changed:
                break
    
    # Verify final result
    if not _check_valid(coloring, adj):
        return list(range(n))
    
    return [int(x) for x in coloring]
