"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

class DSU:
    """Disjoint Set Union with path compression and union by rank."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return True  # cycle detected
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return False

def _has_bicolor_cycle(coloring, n, adj, color_a, color_b):
    """Check if the subgraph induced by vertices with color_a or color_b has a cycle using union-find."""
    vertices = [v for v in range(n) if coloring[v] == color_a or coloring[v] == color_b]
    if len(vertices) < 3:
        return False
    vset = set(vertices)
    # Map vertex to index in DSU
    idx_map = {v: i for i, v in enumerate(vertices)}
    dsu = DSU(len(vertices))
    
    for u in vertices:
        for v in adj[u]:
            if v in vset and u < v:
                if dsu.union(idx_map[u], idx_map[v]):
                    return True
    return False

def _is_valid_assignment(coloring, n, adj, k):
    """Check all constraints: adjacent differ, and no bicolored cycle for any color pair."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = list(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            if _has_bicolor_cycle(coloring, n, adj, colors_used[i], colors_used[j]):
                return False
    return True

def _greedy_mrv(coloring, adj, n, k, deadline):
    """
    Greedy coloring with MRV (Minimum Remaining Values) heuristic.
    Returns True if a valid coloring is found, False otherwise.
    """
    uncolored = set(range(n))
    
    while uncolored:
        if time.time() > deadline:
            return False
        
        # Find vertex with minimum remaining values
        best_v = None
        best_available = None
        best_count = k + 1
        
        for v in uncolored:
            neighbor_colors = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            available = [c for c in range(k) if c not in neighbor_colors]
            count = len(available)
            if count < best_count:
                best_count = count
                best_v = v
                best_available = available
                if count == 0:
                    break
        
        if best_v is None or best_count == 0:
            return False
        
        # Try available colors in random order
        random.shuffle(best_available)
        success = False
        for c in best_available:
            coloring[best_v] = c
            
            # Check if this assignment creates a bicolored cycle
            valid = True
            for nb in adj[best_v]:
                if coloring[nb] != -1 and coloring[nb] != c:
                    ca, cb = c, coloring[nb]
                    # Check bicolored cycle for this pair
                    if _has_bicolor_cycle(coloring, n, adj, ca, cb):
                        valid = False
                        break
            
            if valid:
                uncolored.remove(best_v)
                success = True
                break
            else:
                coloring[best_v] = -1
        
        if not success:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Try to find a valid coloring with minimum colors
    # Use randomized greedy with MRV heuristic
    for k in range(3, 8):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline - 2.0:
                break
            
            coloring = [-1] * n
            
            if _greedy_mrv(coloring, adj, n, k, deadline - 2.0):
                if _is_valid_assignment(coloring, n, adj, k):
                    return [int(x) for x in coloring]
    
    # Fallback: return identity coloring (always valid but uses many colors)
    return list(range(n))

# EVOLVE-BLOCK-END
