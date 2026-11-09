"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55.0
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # DSU class for incremental cycle detection
    class DSU:
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
                return False  # cycle detected
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
    
    # For a given number of colors k, try to find a valid coloring
    def try_coloring(k, max_time=30.0):
        start = time.time()
        
        # We'll use backtracking with MRV heuristic
        # State: color assignment for each vertex (-1 = unassigned)
        # For each pair of colors (c1, c2), maintain a DSU to detect bicolored cycles
        
        # Precompute all color pairs
        color_pairs = list(combinations(range(k), 2))
        
        # For each pair of colors, we need to track the subgraph induced by those two colors
        # A bicolored cycle exists if there's a cycle in the subgraph using only those two colors
        
        # Approach: assign colors one vertex at a time
        # When we assign color c to vertex v, we check all neighbors with colors c' != c
        # For the pair (c, c'), we add edge (v, neighbor) to the bicolored subgraph
        # We need to check if this creates a cycle
        
        # For efficiency, maintain for each color pair a DSU
        # But we only add edges when both endpoints have the same two colors
        
        # Let's think differently: 
        # For each pair (c1, c2), the bicolored subgraph consists of vertices colored c1 or c2,
        # and edges between them. A cycle exists if adding an edge connects two vertices already connected.
        
        # We'll maintain:
        # - color[v] for each vertex v
        # - For each pair (c1, c2), a DSU over the vertices that have color c1 or c2
        #   Actually, we can use a single DSU per pair, but only add edges when both endpoints are assigned
        
        # Simpler approach: after each assignment, check all affected pairs
        # For vertex v assigned color c, for each neighbor u with color c' (c' != c):
        #   The pair (c, c') gets a new edge (v, u)
        #   We need to check if this creates a cycle in the bicolored subgraph for (c, c')
        
        # To check for cycles incrementally, we maintain a DSU for each color pair
        # When we add edge (v, u) to pair (c, c'), we do union(v, u) in DSU[(c,c')]
        # If union fails, there's a cycle
        
        # But we need to be able to undo assignments during backtracking
        # This is complex. Let's use a simpler approach:
        # After each assignment, verify all bicolored subgraphs for affected pairs
        
        # Actually, for 50 vertices and small k, we can just check after each step
        # by building the bicolored subgraph and checking for cycles
        
        def has_bicolored_cycle(color, k):
            # For each pair of colors, check if the induced subgraph has a cycle
            for c1, c2 in combinations(range(k), 2):
                # Get vertices with color c1 or c2
                # Check for cycles in the subgraph induced by these vertices
                # Use DFS/BFS to detect cycles
                
                # Build subgraph
                in_sub = [False] * n
                for v in range(n):
                    if color[v] == c1 or color[v] == c2:
                        in_sub[v] = True
                
                # Check for cycles using DFS
                visited = [False] * n
                parent = [-1] * n
                
                def dfs(u, start):
                    visited[u] = True
                    for v in adj[u]:
                        if not in_sub[v]:
                            continue
                        if not visited[v]:
                            parent[v] = u
                            if dfs(v, start):
                                return True
                        elif v != parent[u]:
                            return True
                    return False
                
                for v in range(n):
                    if in_sub[v] and not visited[v]:
                        if dfs(v, v):
                            return True
            return False
        
        # Backtracking search with MRV
        color = [-1] * n
        
        def is_valid_assignment(v, c):
            # Check if assigning color c to vertex v is valid
            # 1. No neighbor has color c
            for u in adj[v]:
                if color[u] == c:
                    return False
            # 2. Temporarily assign and check for bicolored cycles
            color[v] = c
            valid = not has_bicolored_cycle(color, k)
            color[v] = -1
            return valid
        
        def backtrack(idx):
            if time.time() > start + max_time:
                return False
            if idx == n:
                return True
            
            # MRV: find unassigned vertex with fewest available colors
            min_avail = k + 1
            best_v = -1
            for v in range(idx, n):
                if color[v] == -1:
                    avail = 0
                    for c in range(k):
                        if is_valid_assignment(v, c):
                            avail += 1
                    if avail < min_avail:
                        min_avail = avail
                        best_v = v
                        if avail == 0:
                            return False
                        if avail == 1:
                            break
            
            if best_v == -1:
                return True
            
            # Try colors in order
            for c in range(k):
                if is_valid_assignment(best_v, c):
                    color[best_v] = c
                    if backtrack(idx + 1):
                        return True
                    color[best_v] = -1
            
            return False
        
        if backtrack(0):
            return color
        return None
    
    # Try increasing number of colors
    for k in range(2, 10):
        if time.time() > deadline:
            break
        result = try_coloring(k, max_time=min(25.0, deadline - time.time()))
        if result is not None:
            # Verify the solution
            valid = True
            for u, v in edges:
                if result[u] == result[v]:
                    valid = False
                    break
            if valid:
                # Check bicolored cycles
                for c1, c2 in combinations(range(k), 2):
                    in_sub = [False] * n
                    for v in range(n):
                        if result[v] == c1 or result[v] == c2:
                            in_sub[v] = True
                    
                    visited = [False] * n
                    parent = [-1] * n
                    
                    def dfs(u, start):
                        visited[u] = True
                        for v in adj[u]:
                            if not in_sub[v]:
                                continue
                            if not visited[v]:
                                parent[v] = u
                                if dfs(v, start):
                                    return True
                            elif v != parent[u]:
                                return True
                        return False
                    
                    for v in range(n):
                        if in_sub[v] and not visited[v]:
                            if dfs(v, v):
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                return [int(c) for c in result]
    
    # Fallback: use 50 colors (always valid)
    return list(range(n))

# EVOLVE-BLOCK-END
