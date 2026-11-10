"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Precompute adjacency
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def fast_check_bicolored_cycles(colors):
    """Fast check: for each pair of colors, verify induced subgraph is a forest."""
    k = max(colors) + 1
    for a in range(k):
        for b in range(a + 1, k):
            # For the subgraph induced by vertices with color a or b,
            # check if it's a forest using union-find
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
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a or cu == b) and (cv == a or cv == b):
                    if not union(u, v):
                        return True
    return False

def is_valid(colors):
    """Check all constraints: proper coloring and no bicolored cycles."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    return not fast_check_bicolored_cycles(colors)

def construct():
    deadline = time.time() + 85
    
    # Try systematic patterns first
    # Pattern: i % k
    for k in [3, 4, 5, 6]:
        colors = [i % k for i in range(n)]
        if is_valid(colors):
            return [int(c) for c in colors]
    
    # Try other simple patterns
    # Pattern: (i // 2) % k
    for k in [3, 4, 5, 6]:
        colors = [(i // 2) % k for i in range(n)]
        if is_valid(colors):
            return [int(c) for c in colors]
    
    # Try: i % 3 with some modifications
    # Use smarter local search with fast validity
    def local_search_fast(k, max_iter=10000, seed=None):
        if seed is not None:
            random.seed(seed)
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                return None
            if is_valid(colors):
                return colors
            
            # Pick a random vertex and try all color changes
            v = random.randint(0, n-1)
            old = colors[v]
            best_new = old
            best_cost = float('inf')
            
            for new_c in range(k):
                if new_c == old:
                    continue
                colors[v] = new_c
                if is_valid(colors):
                    colors[v] = old
                    return colors
                # Count violations as a proxy
                colors[v] = old
            
            # Try a random perturbation
            colors[v] = random.randint(0, k-1)
        
        return None
    
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
        for attempt in range(100):
            if time.time() > deadline - 3:
                break
            result = local_search_fast(k, max_iter=3000, seed=random.randint(0, 2**31))
            if result is not None:
                return [int(c) for c in result]
    
    # Ultimate fallback
    return list(range(n))

# EVOLVE-BLOCK-END
