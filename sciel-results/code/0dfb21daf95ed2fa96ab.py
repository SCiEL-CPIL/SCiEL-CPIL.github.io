"""Improved program: backtracking with forward checking for bicolored forest constraints."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 50.0  # Safety margin
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    inv_order = [0] * n
    for i, v in enumerate(order):
        inv_order[v] = i
    
    def solve_with_k(k):
        """Try to find a valid coloring with exactly k colors."""
        if k < 2:
            return None
        
        color = [-1] * n
        # For each pair of colors (c1, c2), maintain DSU to track connectivity
        # in the subgraph induced by vertices with color c1 or c2
        # We'll use a dictionary: (c1, c2) -> DSU state
        
        # Precompute: for each vertex, which colors are forbidden by neighbors
        # We'll use backtracking with forward checking
        
        # DSU implementation for each color pair
        # Since k is small (2-7), we can afford O(k^2) DSUs
        parent = {}
        rank = {}
        
        def dsu_find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        
        def dsu_union(a, b):
            ra, rb = dsu_find(a), dsu_find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        
        def init_dsu():
            nonlocal parent, rank
            parent = {}
            rank = {}
            for i in range(n):
                parent[i] = i
                rank[i] = 0
        
        def can_assign(v, c):
            """Check if vertex v can be assigned color c without creating a bicolored cycle."""
            # Check: no neighbor has color c
            for nb in adj[v]:
                if color[nb] == c:
                    return False
            
            # For each other color c', check if adding v with color c
            # creates a cycle in the {c, c'} subgraph
            # A cycle is created if two neighbors of v with color c' are already connected
            # in the {c, c'} subgraph (considering v as having color c)
            
            for c_prime in range(k):
                if c_prime == c:
                    continue
                
                # Find neighbors of v with color c_prime
                neighbors_with_cp = [nb for nb in adj[v] if color[nb] == c_prime]
                if len(neighbors_with_cp) < 2:
                    continue
                
                # Check if any two of these neighbors are connected in {c, c'} subgraph
                # We need to check connectivity considering only vertices with color c or c_prime
                # This is complex; we'll do a BFS/DFS restricted to {c, c'} subgraph
                
                # For efficiency, we'll check pairwise connectivity
                for i in range(len(neighbors_with_cp)):
                    for j in range(i + 1, len(neighbors_with_cp)):
                        if connected_in_pair(c, c_prime, neighbors_with_cp[i], neighbors_with_cp[j]):
                            return False
            
            return True
        
        def connected_in_pair(c1, c2, u, v):
            """Check if u and v are connected in the subgraph induced by vertices with color c1 or c2."""
            if u == v:
                return True
            
            # BFS from u, only visiting vertices with color c1 or c2
            visited = set()
            queue = deque([u])
            visited.add(u)
            
            while queue:
                cur = queue.popleft()
                if cur == v:
                    return True
                for nb in adj[cur]:
                    if nb in visited:
                        continue
                    if color[nb] == c1 or color[nb] == c2:
                        visited.add(nb)
                        queue.append(nb)
            
            return False
        
        def backtrack(idx):
            """Try to color vertices from idx to n-1 in the ordered sequence."""
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            v = order[idx]
            
            # Try each color
            for c in range(k):
                if can_assign(v, c):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            
            return False
        
        init_dsu()
        return backtrack(0)
    
    # Try increasing number of colors
    best_solution = None
    for k in range(2, n + 1):
        if time.time() > deadline:
            break
        if solve_with_k(k):
            best_solution = color[:]
            break
    
    if best_solution is None:
        # Fallback: trivial solution with all distinct colors
        best_solution = list(range(n))
    
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
