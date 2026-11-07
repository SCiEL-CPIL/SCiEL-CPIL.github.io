"""Rewritten program: minimize distinct colors with bicolored-cycle-free constraint."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolored_cycle(n, edges, coloring, c1, c2):
    """Check if subgraph induced by colors c1 and c2 has a cycle using union-find."""
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
    
    for u, v in edges:
        if coloring[u] == c1 and coloring[v] == c2:
            if union(u, v):
                return True
        elif coloring[u] == c2 and coloring[v] == c1:
            if union(u, v):
                return True
    return False

def _is_valid(n, edges, coloring, num_colors):
    """Check all constraints: proper coloring + no bicolored cycles."""
    # Proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Bicolored cycle check for each pair
    for c1 in range(num_colors):
        for c2 in range(c1 + 1, num_colors):
            if _has_bicolored_cycle(n, edges, coloring, c1, c2):
                return False
    return True

def _try_structural_coloring(n, adj, edges, k, deadline):
    """
    Try to find a valid k-coloring using structural properties of the circulant graph.
    The graph is C(50, {1, 7}) - a circulant graph.
    We try periodic colorings and other structured patterns.
    """
    if time.time() > deadline:
        return None
    
    # Try simple periodic patterns
    for period in range(1, min(n, 20)):
        if time.time() > deadline:
            return None
        # Try all possible base patterns of length 'period'
        base_patterns = []
        for pattern in range(k ** min(period, 3)):  # Limit pattern space
            if period <= 3:
                base = []
                p = pattern
                for i in range(period):
                    base.append(p % k)
                    p //= k
            else:
                base = [i % k for i in range(period)]
            
            coloring = [base[v % period] for v in range(n)]
            
            # Check proper coloring
            valid = True
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check bicolored cycles
            num_used = len(set(coloring))
            if num_used > k:
                continue
            if _is_valid(n, edges, coloring, num_used):
                return coloring
    
    # Try gradient-like patterns
    for start in range(k):
        if time.time() > deadline:
            return None
        coloring = []
        for v in range(n):
            c = (start + v) % k
            coloring.append(c)
        
        valid = True
        for u, v in edges:
            if coloring[u] == coloring[v]:
                valid = False
                break
        if not valid:
            continue
        
        num_used = len(set(coloring))
        if num_used > k:
            continue
        if _is_valid(n, edges, coloring, num_used):
            return coloring
    
    return None

def _backtrack_search(n, adj, edges, k, order, deadline):
    """Backtracking search for k-coloring with bicolored-cycle constraint."""
    coloring = [-1] * n
    
    # Precompute adjacency for quick neighbor color lookup
    def search(idx):
        if time.time() > deadline:
            return None
        if idx == len(order):
            return coloring[:]
        
        v = order[idx]
        
        # Determine forbidden colors from neighbors
        neighbor_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                neighbor_colors.add(coloring[nb])
        
        for c in range(k):
            if c in neighbor_colors:
                continue
            coloring[v] = c
            
            # Check bicolored cycle constraint incrementally
            valid = True
            for nb in adj[v]:
                if coloring[nb] != -1:
                    cn = coloring[nb]
                    if _has_bicolored_cycle(n, edges, coloring, c, cn):
                        valid = False
                        break
            
            if valid:
                result = search(idx + 1)
                if result is not None:
                    return result
            
            coloring[v] = -1
        
        return None
    
    return search(0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    
    # First try structural approaches for small k
    for k in range(3, 6):
        if time.time() > deadline:
            break
        result = _try_structural_coloring(n, adj, edges, k, deadline)
        if result is not None:
            num_used = len(set(result))
            if _is_valid(n, edges, result, num_used):
                return [int(c) for c in result]
    
    # Vertex ordering: degree descending, then by index for determinism
    degrees = [len(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: (-degrees[v], v))
    
    # Try backtracking for k = 3, 4, 5, ...
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = _backtrack_search(n, adj, edges, k, order, deadline)
        if result is not None:
            num_used = len(set(result))
            if _is_valid(n, edges, result, num_used):
                return [int(c) for c in result]
    
    # Fallback: greedy coloring (should always work with enough colors)
    coloring = [0] * n
    for v in range(n):
        neighbor_colors = set(coloring[nb] for nb in adj[v])
        c = 0
        while c in neighbor_colors:
            c += 1
        coloring[v] = c
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
